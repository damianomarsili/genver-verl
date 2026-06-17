# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BOX_COLOR = (0, 0, 255)
FONT_SCALE = 0.022
BOX_OUTLINE_SCALE = 0.005
LABEL_PADDING = 2
VERIFIER_IMAGE_JPEG_QUALITY = 85
BBOX_2D_ENTRY_PATTERN = re.compile(
    r'^\s*(?P<idx>\d+)\s*:\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\[(?P<coords>[^\]]+)\]\s*$',
    flags=re.IGNORECASE,
)
VERIFIER_EDIT_PATTERN = re.compile(r"(?i)^EDIT\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
VERIFIER_REMOVE_PATTERN = re.compile(r"(?i)^REMOVE\s+(?P<idx>\d+)\s*$")
VERIFIER_ADD_PATTERN = re.compile(
    r'(?i)^ADD\s+label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\['
    r"\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*$"
)
VERIFIER_COORD_KEYWORD_PATTERN = re.compile(r"(?i)\b(x_min|x_max|y_min|y_max|left|right|top|bottom|width|height)\b")
VERIFIER_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class LocEntry:
    image_index: int
    label: str
    coords: Tuple[float, ...]


@register("genver_agent")
class GenVerAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        verifier_image_side: int = 1024,
        loc_verifier_rounds: int = 3,
        verifier_max_new_tokens: int = 96,
        max_new_tokens_per_chunk: int = 256,
        **kwargs: Any,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        self.instruction_mode = instruction_mode.lower()
        if self.instruction_mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")

        self.max_image_side = int(max_image_side)
        self.verifier_image_side = int(verifier_image_side)
        self.loc_verifier_rounds = max(0, int(loc_verifier_rounds))
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        genver_perf_cfg = config.algorithm.get("genver_perf", {}) if "algorithm" in config else {}
        self.aux_mm_reuse_enable = self._coerce_bool(genver_perf_cfg.get("aux_mm_reuse_enable", True), True)
        self.agent_loop_cpu_cleanup_enable = self._coerce_bool(
            genver_perf_cfg.get("agent_loop_cpu_cleanup_enable", True), True
        )

        self.max_new_tokens_per_chunk = int(max_new_tokens_per_chunk)

        prompts_dir = self._resolve_prompts_dir()
        prompt_file = prompts_dir / "genver_verifier_single_turn.txt"
        self.prompt_template = self._read_prompt_file(prompt_file)
        self.instruction_text = self._read_prompt_file(prompts_dir / "instructions_box.txt")
        self.verifier_template = self._read_prompt_file(prompts_dir / "verifier_instructions.txt")

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return default

    def _resolve_prompts_dir(self) -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "training" / "prompts").exists():
                return parent / "training" / "prompts"
        raise FileNotFoundError("Could not locate training/prompts relative to the repo.")

    def _read_prompt_file(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        return text.replace("<plan>", "<reason>").replace("</plan>", "</reason>")

    def _build_prompted_context(self, query: str) -> str:
        return self.prompt_template.format(self.instruction_text, query.strip())

    def _resize_longest_side(self, image: Image.Image, longest_side: int) -> Image.Image:
        width, height = image.size
        if max(width, height) <= longest_side:
            return image
        if width >= height:
            new_width = longest_side
            new_height = int(round(height * (longest_side / width)))
        else:
            new_height = longest_side
            new_width = int(round(width * (longest_side / height)))
        return image.resize((new_width, new_height), Image.BICUBIC)

    def _coerce_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and "bytes" in image:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _serialize_image_bytes(self, image: Image.Image) -> dict[str, Any]:
        buffer = BytesIO()
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=VERIFIER_IMAGE_JPEG_QUALITY,
            optimize=True,
        )
        return {
            "bytes": buffer.getvalue(),
            "format": "JPEG",
            "width": image.width,
            "height": image.height,
        }

    def _normalize_images(self, images: Optional[list[Any]]) -> list[Image.Image]:
        if not images:
            return []
        return [self._coerce_pil(img) for img in images]

    def _build_messages(self, prompt: str, images: list[Image.Image]) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for image in images:
            # `images` are expected to be pre-resized RGB images.
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _extract_query(self, messages: list[dict[str, object]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            text = self._extract_text_from_content(content)
            if text:
                return text.strip()
        return ""

    def _extract_text_from_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return ""

    def _load_font(self, image: Image.Image, scale: float = FONT_SCALE) -> ImageFont.ImageFont:
        width, height = image.size
        size = max(12, int(min(width, height) * scale))
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _scale_point(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        x_px = int(round(x / 1000.0 * width))
        y_px = int(round(y / 1000.0 * height))
        return max(0, min(width - 1, x_px)), max(0, min(height - 1, y_px))

    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        if hasattr(font, "getbbox"):
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        return font.getsize(text)

    def _overlay_boxes(self, image: Image.Image, entries: list[LocEntry]) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(base)

        width, height = base.size
        outline_width = max(2, int(min(width, height) * BOX_OUTLINE_SCALE))
        indexed_entries: list[tuple[int, LocEntry, float]] = []
        for idx, entry in enumerate(entries, start=1):
            x1, y1, x2, y2 = entry.coords[:4]
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            indexed_entries.append((idx, entry, area))
        # Draw larger boxes first so smaller nested boxes remain visible on top.
        indexed_entries.sort(key=lambda item: item[2], reverse=True)

        for idx, entry, _ in indexed_entries:
            x1, y1, x2, y2 = entry.coords[:4]
            left, top = self._scale_point(x1, y1, width, height)
            right, bottom = self._scale_point(x2, y2, width, height)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            draw.rectangle(
                (left, top, right, bottom),
                outline=BOX_COLOR,
                width=outline_width,
            )
            label = f"{idx}:{entry.label}" if entry.label else str(idx)
            text_w, text_h = self._measure_text(draw, label, font)
            text_x = min(width - text_w - LABEL_PADDING, max(0, left + LABEL_PADDING))
            text_y = min(height - text_h - LABEL_PADDING, max(0, top + LABEL_PADDING))
            draw.rectangle(
                (text_x - LABEL_PADDING, text_y - LABEL_PADDING, text_x + text_w + LABEL_PADDING, text_y + text_h),
                fill=(255, 255, 255, 200),
            )
            draw.text((text_x, text_y), label, fill=BOX_COLOR, font=font)

        return Image.alpha_composite(base, overlay).convert("RGB")

    def _extract_bbox_2d_payloads(self, text: str) -> list[str]:
        if not text:
            return []
        payloads: list[str] = []
        for raw_payload in re.findall(r"(?is)<bbox_2d>(.*?)</bbox_2d>", text):
            payloads.append(str(raw_payload).strip())
        return payloads

    def _extract_loc_token_spans(
        self,
        chunk: str,
        *,
        chunk_start: int,
        chunk_token_count: int,
    ) -> list[dict[str, Any]]:
        """Extract token spans for <bbox_2d> calls in one assistant chunk.

        Uses char-level span to token-level conversion; falls back to chunk-level
        when an exact match cannot be established.
        """
        spans: list[dict[str, Any]] = []
        if chunk_token_count <= 0:
            return spans

        def _encode_len(text: str) -> int:
            if not text:
                return 0
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        loc_matches = list(re.finditer(r"(?is)<bbox_2d>.*?</bbox_2d>", chunk))
        if not loc_matches and "<bbox_2d>" in chunk.lower():
            loc_matches = [None]

        if not loc_matches:
            return spans

        # Strict single-block format: if multiple blocks appear in one chunk,
        # treat them as one invalid span for one tool-call record.
        if len(loc_matches) > 1:
            first = loc_matches[0]
            last = loc_matches[-1]
            prefix = chunk[: first.start()]
            loc_text = chunk[first.start() : last.end()]
            prefix_tokens = _encode_len(prefix)
            loc_tokens = _encode_len(loc_text)
            token_start = chunk_start + prefix_tokens
            token_end = token_start + max(1, loc_tokens)
            chunk_end = chunk_start + chunk_token_count
            if token_start >= chunk_end:
                token_start = chunk_start
                token_end = chunk_end
                span_type = "chunk_fallback"
            else:
                token_end = min(token_end, chunk_end)
                span_type = "merged_multi_block"
            spans.append(
                {
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": loc_text,
                    "span_type": span_type,
                }
            )
            return spans

        for match in loc_matches:
            if match is None:
                token_start = chunk_start
                token_end = chunk_start + chunk_token_count
                spans.append(
                    {
                        "token_start": token_start,
                        "token_end": token_end,
                        "text": chunk,
                        "span_type": "chunk_fallback",
                    }
                )
                continue

            prefix = chunk[: match.start()]
            loc_text = chunk[match.start() : match.end()]
            prefix_tokens = _encode_len(prefix)
            loc_tokens = _encode_len(loc_text)
            token_start = chunk_start + prefix_tokens
            token_end = token_start + max(1, loc_tokens)

            chunk_end = chunk_start + chunk_token_count
            if token_start >= chunk_end:
                token_start = chunk_start
                token_end = chunk_end
                span_type = "chunk_fallback"
            else:
                token_end = min(token_end, chunk_end)
                span_type = "exact"

            spans.append(
                {
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": loc_text,
                    "span_type": span_type,
                }
            )

        return spans

    def _parse_bbox_2d_entries(self, payload: str) -> list[LocEntry]:
        entries: list[LocEntry] = []
        nonempty_line_count = 0
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            nonempty_line_count += 1
            match = BBOX_2D_ENTRY_PATTERN.fullmatch(line)
            if match is None:
                return []
            try:
                idx = int(match.group("idx"))
            except (TypeError, ValueError):
                return []
            if idx != len(entries) + 1:
                return []

            label = match.group("label").strip()
            if not label:
                return []

            numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("coords"))
            if len(numbers) != 4:
                return []
            coords_tuple = tuple(float(n) for n in numbers[:4])
            x1, y1, x2, y2 = coords_tuple
            if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
                return []
            if x2 <= x1 or y2 <= y1:
                return []

            entries.append(LocEntry(image_index=1, label=label, coords=coords_tuple))
        if nonempty_line_count == 0:
            return []
        return entries

    def _has_missing_label(self, payload: str) -> bool:
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            has_coords = bool(re.search(r"\[[^\]]*\]", line))
            has_label = bool(re.search(r'(?i)\blabel\s*=\s*"', line))
            if has_coords and not has_label:
                return True
        return False

    def _has_invalid_box(self, entries: list[LocEntry]) -> bool:
        for entry in entries:
            if len(entry.coords) < 4:
                continue
            x1, y1, x2, y2 = entry.coords[:4]
            if x2 <= x1 or y2 <= y1:
                return True
            if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
                return True
        return False

    def _build_verifier_prompt(self, entries: list[LocEntry], image_count: int) -> str:
        targets = sorted({entry.label for entry in entries})
        targets_str = ", ".join(targets) if targets else "(none)"

        lines: list[str] = []
        for i, entry in enumerate(entries, 1):
            prefix = f"image_{entry.image_index}, " if image_count > 1 else ""
            x1, y1, x2, y2 = entry.coords[:4]
            lines.append(f'{i}) {prefix}label="{entry.label}", [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]')

        preds_str = "\n".join(lines) if lines else "(none)"
        return self.verifier_template.format(targets=targets_str, preds=preds_str)

    def _parse_verifier_feedback(self, text: str, entries: list[LocEntry]) -> tuple[str, str, dict[str, Any]]:
        cleaned = text.replace("<|im_end|>", "").strip()
        valid_indices = set(range(1, len(entries) + 1))
        index_actions: dict[int, tuple[str, int]] = {}
        remove_actions: dict[int, tuple[str, int]] = {}
        raw_add_actions: list[tuple[str, int, tuple[str, float, float, float, float]]] = []
        line_order = 0

        def _format_coord(value: float) -> str:
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.3f}".rstrip("0").rstrip(".")

        def _normalize_label(label: str) -> str:
            return " ".join(str(label).strip().lower().split())

        def _canonical_signature(
            *,
            label: str,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
        ) -> tuple[str, float, float, float, float]:
            return (
                _normalize_label(label),
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2), 3),
                round(float(y2), 3),
            )

        existing_signatures: set[tuple[str, float, float, float, float]] = set()
        for entry in entries:
            x1, y1, x2, y2 = entry.coords[:4]
            signature = _canonical_signature(label=entry.label, x1=x1, y1=y1, x2=x2, y2=y2)
            existing_signatures.add(signature)

        duplicate_add_existing_count = 0
        disallowed_remove_count = 0
        invalid_remove_count = 0
        remove_add_duplicate_count = 0
        removed_signatures: set[tuple[str, float, float, float, float]] = set()

        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue
            if line.lower().startswith("corrections:"):
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue

            edit_match = VERIFIER_EDIT_PATTERN.match(line)
            if edit_match is not None:
                idx = int(edit_match.group("idx"))
                body = edit_match.group("body").strip()
                has_coord_keyword = VERIFIER_COORD_KEYWORD_PATTERN.search(body) is not None
                has_numeric_value = VERIFIER_NUMBER_PATTERN.search(body) is not None
                if idx in valid_indices and body and has_coord_keyword and has_numeric_value:
                    index_actions[idx] = (f"EDIT {idx}: {body}", line_order)
                    line_order += 1
                continue

            remove_match = VERIFIER_REMOVE_PATTERN.match(line)
            if remove_match is not None:
                try:
                    idx = int(remove_match.group("idx"))
                except (TypeError, ValueError):
                    invalid_remove_count += 1
                    continue
                if idx not in valid_indices:
                    invalid_remove_count += 1
                    continue
                if idx not in remove_actions:
                    remove_actions[idx] = (f"REMOVE {idx}", line_order)
                    line_order += 1
                    removed_entry = entries[idx - 1]
                    rx1, ry1, rx2, ry2 = removed_entry.coords[:4]
                    removed_signatures.add(
                        _canonical_signature(
                            label=removed_entry.label,
                            x1=rx1,
                            y1=ry1,
                            x2=rx2,
                            y2=ry2,
                        )
                    )
                continue

            add_match = VERIFIER_ADD_PATTERN.match(line)
            if add_match is not None:
                label = add_match.group("label").strip()
                if not label:
                    continue
                x1 = float(add_match.group("x1"))
                y1 = float(add_match.group("y1"))
                x2 = float(add_match.group("x2"))
                y2 = float(add_match.group("y2"))
                if not (
                    0.0 <= x1 <= 1000.0
                    and 0.0 <= y1 <= 1000.0
                    and 0.0 <= x2 <= 1000.0
                    and 0.0 <= y2 <= 1000.0
                    and x2 > x1
                    and y2 > y1
                ):
                    continue
                normalized = (
                    f'ADD label="{label}", '
                    f"[{_format_coord(x1)}, {_format_coord(y1)}, {_format_coord(x2)}, {_format_coord(y2)}]"
                )
                signature = _canonical_signature(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
                if signature in removed_signatures:
                    # Contradictory instruction: removing then re-adding the same box.
                    remove_add_duplicate_count += 1
                    continue
                raw_add_actions.append((normalized, line_order, signature))
                line_order += 1

        add_actions: list[tuple[str, int]] = []
        add_seen: set[tuple[str, float, float, float, float]] = set()
        for line, order, signature in raw_add_actions:
            if signature in existing_signatures:
                duplicate_add_existing_count += 1
                continue
            if signature in add_seen:
                continue
            add_seen.add(signature)
            add_actions.append((line, order))

        normalized_lines: list[tuple[str, int]] = (
            list(index_actions.values()) + list(remove_actions.values()) + add_actions
        )
        normalized_lines.sort(key=lambda item: item[1])
        has_effect = len(normalized_lines) > 0
        feedback_valid_for_reward = (
            has_effect
            and duplicate_add_existing_count == 0
            and disallowed_remove_count == 0
            and invalid_remove_count == 0
            and remove_add_duplicate_count == 0
        )
        feedback_info = {
            "feedback_has_effect": bool(has_effect),
            "feedback_valid_for_reward": bool(feedback_valid_for_reward),
            "feedback_has_duplicate_add_existing": bool(duplicate_add_existing_count > 0),
            "feedback_has_disallowed_remove": bool(disallowed_remove_count > 0),
            "feedback_has_invalid_remove": bool(invalid_remove_count > 0),
            "feedback_has_remove_add_duplicate": bool(remove_add_duplicate_count > 0),
            "feedback_duplicate_add_existing_count": int(duplicate_add_existing_count),
            "feedback_disallowed_remove_count": int(disallowed_remove_count),
            "feedback_invalid_remove_count": int(invalid_remove_count),
            "feedback_remove_add_duplicate_count": int(remove_add_duplicate_count),
        }
        if len(normalized_lines) == 0:
            no_op = "NO_VALID_CORRECTIONS. Re-emit all boxes unchanged in one <bbox_2d> block."
            return no_op, cleaned, feedback_info
        corrections = "\n".join(line for line, _ in normalized_lines)
        return corrections, cleaned, feedback_info

    async def _generate_once(
        self,
        messages: list[dict[str, object]],
        images: list[Image.Image],
        sampling_params: dict[str, Any],
        stop_sequences: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list[int], list[float]]:
        chunk, token_ids, log_probs, _ = await self._generate_once_with_prompt_ids(
            messages=messages,
            images=images,
            sampling_params=sampling_params,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
            metrics=metrics,
        )
        return chunk, token_ids, log_probs

    async def _generate_once_with_prompt_ids(
        self,
        messages: list[dict[str, object]],
        images: list[Image.Image],
        sampling_params: dict[str, Any],
        stop_sequences: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list[int], list[float], list[int]]:
        prompt_ids = await self.apply_chat_template(messages, images=images)
        if stop_sequences:
            sampling_params = dict(sampling_params)
            sampling_params["stop"] = stop_sequences
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = max_new_tokens
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
        )
        chunk = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        return chunk, output.token_ids, log_probs, prompt_ids

    async def _append_user_message(
        self,
        messages: list[dict[str, object]],
        text: str,
        train_prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]],
        images: Optional[list[Image.Image]] = None,
    ) -> list[int]:
        add_messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        messages.extend(add_messages)
        extra_prompt_ids = await self.apply_chat_template(
            add_messages, remove_system_prompt=True, images=None
        )
        train_prompt_ids.extend(extra_prompt_ids)
        response_mask.extend([0] * len(extra_prompt_ids))
        if response_logprobs is not None:
            response_logprobs.extend([0.0] * len(extra_prompt_ids))
        return extra_prompt_ids

    async def _generate_once_with_cached_prompt_ids(
        self,
        prompt_ids: list[int],
        images: list[Image.Image],
        sampling_params: dict[str, Any],
        stop_sequences: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list[int], list[float]]:
        if stop_sequences:
            sampling_params = dict(sampling_params)
            sampling_params["stop"] = stop_sequences
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = max_new_tokens
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
        )
        chunk = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        return chunk, output.token_ids, log_probs

    def _append_assistant_tokens(
        self,
        output_chunks: list[str],
        chunk: str,
        token_ids: list[int],
        train_prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]],
        log_probs: list[float],
    ) -> int:
        output_chunks.append(chunk)
        train_prompt_ids.extend(token_ids)
        response_mask.extend([1] * len(token_ids))
        if response_logprobs is not None:
            if log_probs:
                response_logprobs.extend(log_probs)
            else:
                response_logprobs.extend([0.0] * len(token_ids))
        return len(token_ids)

    def _format_bbox_block(self, entries: list[LocEntry]) -> str:
        def _format_coord(value: float) -> str:
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.3f}".rstrip("0").rstrip(".")

        lines: list[str] = []
        for idx, entry in enumerate(entries, start=1):
            x1, y1, x2, y2 = entry.coords[:4]
            lines.append(
                f'{idx}: label="{entry.label}", '
                f'[{_format_coord(x1)}, {_format_coord(y1)}, {_format_coord(x2)}, {_format_coord(y2)}]'
            )
        return "<bbox_2d>\n" + "\n".join(lines) + "\n</bbox_2d>"

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str) -> str:
        query_text = query.strip()
        return (
            f"{self.instruction_text}\n\n"
            f"Original query:\n{query_text}\n\n"
            f"Detected objects:\n{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags and then putting ONLY your final "
            "answer inside <answer>. Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    async def _build_prompt_ids_and_mm_inputs(
        self,
        messages: list[dict[str, object]],
        images: list[Image.Image],
    ) -> tuple[list[int], dict[str, torch.Tensor]]:
        # Keep aux-generation prompt_ids identical across modes: prompt_ids always come
        # from the standard chat-template tokenizer path.
        prompt_ids = await self.apply_chat_template(messages, images=images)
        if self.processor is None or not self.aux_mm_reuse_enable:
            return prompt_ids, {}

        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        model_inputs = self.processor(
            text=[raw_prompt],
            images=images,
            return_tensors="pt",
            do_sample_frames=False,
        )
        model_inputs.pop("input_ids", None)
        model_inputs.pop("attention_mask", None)
        multi_modal_inputs = dict(model_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        normalized: dict[str, torch.Tensor] = {}
        for key, value in multi_modal_inputs.items():
            if isinstance(value, torch.Tensor):
                normalized[key] = value.detach().cpu().contiguous()
        return prompt_ids, normalized

    async def _build_verifier_messages(
        self,
        originals: list[Image.Image],
        overlays: list[Image.Image],
        prompt: str,
        generate_logprobs: bool,
        metrics: dict[str, Any],
    ) -> Tuple[str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, torch.Tensor]]:
        resized_originals = [self._resize_longest_side(img, self.verifier_image_side) for img in originals]
        resized_overlays = [self._resize_longest_side(img, self.verifier_image_side) for img in overlays]

        content: list[dict[str, object]] = []
        verifier_images: list[Image.Image] = []
        for original, overlay in zip(resized_originals, resized_overlays, strict=True):
            content.append({"type": "image", "image": original})
            content.append({"type": "image", "image": overlay})
            verifier_images.extend([original, overlay])
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        prompt_ids, verifier_multi_modal_inputs = await self._build_prompt_ids_and_mm_inputs(
            messages,
            verifier_images,
        )

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": bool(generate_logprobs),
            "max_tokens": int(self.verifier_max_new_tokens),
        }
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=verifier_images,
        )
        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        output_ids = output.token_ids
        output_log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output_log_probs:
            verifier_output_log_probs = [float(x) for x in output_log_probs]
        else:
            verifier_output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in verifier_images]
        return (
            text,
            prompt_ids,
            output_ids,
            verifier_output_log_probs,
            serialized_images,
            verifier_multi_modal_inputs,
        )

    async def _build_answer_aux_messages(
        self,
        images: list[Image.Image],
        query: str,
        latest_bbox_block: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[str, str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, torch.Tensor]]:
        answer_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        messages = self._build_messages(answer_prompt, images)
        prompt_ids, answer_multi_modal_inputs = await self._build_prompt_ids_and_mm_inputs(messages, images)
        answer_sampling_params = dict(sampling_params)
        answer_sampling_params["max_tokens"] = self.max_new_tokens_per_chunk
        answer_sampling_params["stop"] = ["</answer>"]
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=answer_sampling_params,
            image_data=images,
        )
        output_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        output_ids = output.token_ids
        output_log_probs_raw = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output_log_probs_raw:
            output_log_probs = [float(x) for x in output_log_probs_raw]
        else:
            output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in images]
        return (
            answer_prompt,
            output_text,
            prompt_ids,
            output_ids,
            output_log_probs,
            serialized_images,
            answer_multi_modal_inputs,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        validate_mode = bool(kwargs.get("validate", False))
        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        raw_images_rgb = [img.convert("RGB") for img in raw_images]
        # Resize images for model input so prompt/image token length reflects GENVER_MAX_IMAGE_SIDE.
        images = [self._resize_longest_side(img, self.max_image_side) for img in raw_images_rgb]

        prompted_query = self._build_prompted_context(query)
        messages = self._build_messages(prompted_query, images)
        initial_prompt_ids = await self.apply_chat_template(messages, images=images)
        model_prompt_ids = list(initial_prompt_ids)

        train_prompt_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = [] if sampling_params.get("logprobs") else None
        genver_answer_mask: list[int] = []
        genver_loc_mask: list[int] = []
        genver_loc_calls: list[dict[str, Any]] = []
        genver_loc_verifier_calls: list[dict[str, Any]] = []
        genver_answer_aux_call: Optional[dict[str, Any]] = None
        loc_call_counter = 0

        output_chunks: list[str] = []
        max_total_tokens = self.response_length
        total_generated_tokens = 0
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'

        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1}
        user_turns = 1
        assistant_turns = 0

        def _extend_objective_masks(answer_value: int, loc_value: int, count: int) -> None:
            if count <= 0:
                return
            genver_answer_mask.extend([answer_value] * count)
            genver_loc_mask.extend([loc_value] * count)

        async def _append_user_turn(text: str) -> None:
            extra_prompt_ids = await self._append_user_message(
                messages,
                text,
                train_prompt_ids,
                response_mask,
                response_logprobs,
            )
            if self.agent_loop_cpu_cleanup_enable and extra_prompt_ids:
                model_prompt_ids.extend(extra_prompt_ids)
            appended = len(extra_prompt_ids)
            _extend_objective_masks(answer_value=0, loc_value=0, count=appended)

        def _append_assistant_turn(chunk: str, token_ids: list[int], log_probs: list[float]) -> None:
            nonlocal loc_call_counter
            chunk_start = len(response_mask)
            appended = self._append_assistant_tokens(
                output_chunks,
                chunk,
                token_ids,
                train_prompt_ids,
                response_mask,
                response_logprobs,
                log_probs,
            )
            # Match no-verifier semantics: assistant tokens are answer-objective by default.
            # Then explicitly remove non-answer spans below.
            _extend_objective_masks(answer_value=1, loc_value=0, count=appended)

            for loc_span in self._extract_loc_token_spans(
                chunk,
                chunk_start=chunk_start,
                chunk_token_count=appended,
            ):
                span_start = max(0, int(loc_span["token_start"]))
                span_end = min(len(genver_loc_mask), int(loc_span["token_end"]))
                if span_end <= span_start:
                    continue
                # Decouple objectives: do not optimize answer loss on bbox tokens.
                genver_answer_mask[span_start:span_end] = [0] * (span_end - span_start)
                genver_loc_mask[span_start:span_end] = [1] * (span_end - span_start)
                loc_text = str(loc_span.get("text", chunk))
                parsed_entries_1000: list[dict[str, Any]] = []
                payloads = self._extract_bbox_2d_payloads(loc_text)
                if len(payloads) == 1:
                    parsed_entries = self._parse_bbox_2d_entries(payloads[0])
                    if len(parsed_entries) > 0:
                        parsed_entries_1000 = [
                            {
                                "image_index": int(entry.image_index),
                                "label": str(entry.label),
                                "box_1000": [
                                    float(entry.coords[0]),
                                    float(entry.coords[1]),
                                    float(entry.coords[2]),
                                    float(entry.coords[3]),
                                ],
                            }
                            for entry in parsed_entries
                        ]
                genver_loc_calls.append(
                    {
                        "call_index": loc_call_counter,
                        "token_start": span_start,
                        "token_end": span_end,
                        "text": loc_text,
                        "span_type": loc_span.get("span_type", "chunk_fallback"),
                        "parsed_entries_1000": parsed_entries_1000,
                    }
                )
                loc_call_counter += 1

        def _return_output() -> AgentLoopOutput:
            response_ids = train_prompt_ids[len(initial_prompt_ids) :]
            max_response_len = min(len(response_ids), self.response_length)
            clipped_loc_calls: list[dict[str, Any]] = []
            for call in genver_loc_calls:
                token_start = max(0, int(call.get("token_start", 0)))
                token_end = min(max_response_len, int(call.get("token_end", 0)))
                if token_end <= token_start:
                    continue
                clipped = dict(call)
                clipped["token_start"] = token_start
                clipped["token_end"] = token_end
                clipped_loc_calls.append(clipped)
            return AgentLoopOutput(
                prompt_ids=initial_prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=(response_logprobs[: self.response_length] if response_logprobs is not None else None),
                multi_modal_data={"images": images} if images else {},
                num_turns=user_turns + assistant_turns,
                metrics=metrics,
                extra_fields={
                    "genver_answer_mask": genver_answer_mask[: self.response_length],
                    "genver_loc_mask": genver_loc_mask[: self.response_length],
                    "genver_loc_calls": clipped_loc_calls,
                    "genver_loc_verifier_calls": genver_loc_verifier_calls,
                    "genver_answer_aux_call": genver_answer_aux_call,
                },
            )

        while True:
            if len(response_mask) >= self.response_length:
                return _return_output()

            if self.agent_loop_cpu_cleanup_enable:
                chunk, token_ids, log_probs = await self._generate_once_with_cached_prompt_ids(
                    prompt_ids=model_prompt_ids,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
            else:
                chunk, token_ids, log_probs = await self._generate_once(
                    messages,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
            if not token_ids and not chunk.strip():
                return _return_output()

            total_generated_tokens += len(token_ids)
            assistant_turns += 1
            _append_assistant_turn(chunk, token_ids, log_probs)
            if self.agent_loop_cpu_cleanup_enable and token_ids:
                model_prompt_ids.extend(token_ids)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

            if total_generated_tokens >= max_total_tokens:
                return _return_output()
            if "</answer>" in chunk:
                return _return_output()

            loc_payloads = self._extract_bbox_2d_payloads(chunk)
            if len(loc_payloads) != 1:
                return _return_output()
            loc_payload = loc_payloads[0]

            if self._has_missing_label(loc_payload):
                return _return_output()

            entries = self._parse_bbox_2d_entries(loc_payload)
            if not entries:
                return _return_output()
            if self._has_invalid_box(entries):
                return _return_output()

            current_entries = entries
            for verifier_round in range(self.loc_verifier_rounds):
                entries_by_image: dict[int, list[LocEntry]] = {}
                for entry in current_entries:
                    entries_by_image.setdefault(entry.image_index, []).append(entry)

                originals: list[Image.Image] = []
                overlays: list[Image.Image] = []
                for image_idx in range(1, len(raw_images_rgb) + 1):
                    original = raw_images_rgb[image_idx - 1]
                    image_entries = entries_by_image.get(image_idx, [])
                    overlay = self._overlay_boxes(original, image_entries)
                    originals.append(original)
                    overlays.append(overlay)

                verifier_prompt = self._build_verifier_prompt(current_entries, len(images))
                (
                    verifier_output,
                    verifier_prompt_ids,
                    verifier_output_ids,
                    verifier_output_log_probs,
                    verifier_images,
                    verifier_multi_modal_inputs,
                ) = await self._build_verifier_messages(
                    originals,
                    overlays,
                    verifier_prompt,
                    generate_logprobs=bool(sampling_params.get("logprobs", False)),
                    metrics=metrics,
                )
                corrections, _, feedback_info = self._parse_verifier_feedback(verifier_output, current_entries)
                current_loc_call_index = genver_loc_calls[-1]["call_index"] if genver_loc_calls else -1
                if current_loc_call_index >= 0:
                    genver_loc_verifier_calls.append(
                        {
                            "call_index": int(current_loc_call_index),
                            "round_index": int(verifier_round),
                            "corrections": str(corrections),
                            "verifier_prompt_text": verifier_prompt,
                            "verifier_output_text": verifier_output,
                            "verifier_prompt_token_ids": verifier_prompt_ids,
                            "verifier_output_token_ids": verifier_output_ids,
                            "verifier_output_log_probs": verifier_output_log_probs,
                            "verifier_images": verifier_images,
                            "verifier_multi_modal_inputs": verifier_multi_modal_inputs,
                            "genver_loc_verifier_feedback_has_effect": bool(
                                feedback_info.get("feedback_has_effect", False)
                            ),
                            "genver_loc_verifier_feedback_valid_for_reward": bool(
                                feedback_info.get("feedback_valid_for_reward", False)
                            ),
                            "genver_loc_verifier_feedback_has_duplicate_add_existing": bool(
                                feedback_info.get("feedback_has_duplicate_add_existing", False)
                            ),
                            "genver_loc_verifier_feedback_has_disallowed_remove": bool(
                                feedback_info.get("feedback_has_disallowed_remove", False)
                            ),
                            "genver_loc_verifier_feedback_has_invalid_remove": bool(
                                feedback_info.get("feedback_has_invalid_remove", False)
                            ),
                            "genver_loc_verifier_feedback_has_remove_add_duplicate": bool(
                                feedback_info.get("feedback_has_remove_add_duplicate", False)
                            ),
                            "genver_loc_verifier_feedback_duplicate_add_existing_count": int(
                                feedback_info.get("feedback_duplicate_add_existing_count", 0)
                            ),
                            "genver_loc_verifier_feedback_disallowed_remove_count": int(
                                feedback_info.get("feedback_disallowed_remove_count", 0)
                            ),
                            "genver_loc_verifier_feedback_invalid_remove_count": int(
                                feedback_info.get("feedback_invalid_remove_count", 0)
                            ),
                            "genver_loc_verifier_feedback_remove_add_duplicate_count": int(
                                feedback_info.get("feedback_remove_add_duplicate_count", 0)
                            ),
                        }
                    )

                await _append_user_turn(
                    (
                        "I have some feedback for you to incorporate. "
                        f"Please output ONLY one <bbox_2d> block using lines formatted as {bbox_line_format} "
                        "that incorporates the feedback.\n"
                        f"Feedback: {corrections}\n"
                        "You MUST re-predict ALL boxes, including unchanged ones, and keep indices sequential "
                        "starting at 1. You MUST incorporate the feedback and MUST NOT make unrelated changes."
                    )
                )

                if self.agent_loop_cpu_cleanup_enable:
                    correction_chunk, correction_token_ids, correction_log_probs = (
                        await self._generate_once_with_cached_prompt_ids(
                            prompt_ids=model_prompt_ids,
                            images=images,
                            sampling_params=sampling_params,
                            stop_sequences=["</bbox_2d>"],
                            max_new_tokens=self.max_new_tokens_per_chunk,
                            metrics=metrics,
                        )
                    )
                else:
                    correction_chunk, correction_token_ids, correction_log_probs = await self._generate_once(
                        messages,
                        images=images,
                        sampling_params=sampling_params,
                        stop_sequences=["</bbox_2d>"],
                        max_new_tokens=self.max_new_tokens_per_chunk,
                        metrics=metrics,
                    )
                total_generated_tokens += len(correction_token_ids)
                assistant_turns += 1
                _append_assistant_turn(correction_chunk, correction_token_ids, correction_log_probs)
                if self.agent_loop_cpu_cleanup_enable and correction_token_ids:
                    model_prompt_ids.extend(correction_token_ids)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": correction_chunk}]})

                corrected_payloads = self._extract_bbox_2d_payloads(correction_chunk)
                if len(corrected_payloads) != 1:
                    continue
                corrected_payload = corrected_payloads[0]
                if self._has_missing_label(corrected_payload):
                    continue
                corrected_entries = self._parse_bbox_2d_entries(corrected_payload)
                if not corrected_entries:
                    continue
                if self._has_invalid_box(corrected_entries):
                    continue
                current_entries = corrected_entries

            latest_bbox_block = self._format_bbox_block(current_entries)
            (
                answer_prompt_text,
                answer_output_text,
                answer_prompt_token_ids,
                answer_output_token_ids,
                answer_output_log_probs,
                answer_images,
                answer_multi_modal_inputs,
            ) = await self._build_answer_aux_messages(
                images=images,
                query=query,
                latest_bbox_block=latest_bbox_block,
                sampling_params=sampling_params,
                metrics=metrics,
            )
            genver_answer_aux_call = {
                "answer_prompt_text": answer_prompt_text,
                "answer_output_text": answer_output_text,
                "answer_prompt_token_ids": answer_prompt_token_ids,
                "answer_output_token_ids": answer_output_token_ids,
                "answer_output_log_probs": answer_output_log_probs,
                "answer_images": answer_images,
                "answer_multi_modal_inputs": answer_multi_modal_inputs,
                "answer_latest_bbox_block": latest_bbox_block,
                "answer_solution_str": f"{latest_bbox_block}\n{answer_output_text}",
            }
            if validate_mode:
                total_generated_tokens += len(answer_output_token_ids)
                assistant_turns += 1
                _append_assistant_turn(answer_output_text, answer_output_token_ids, answer_output_log_probs)
                if self.agent_loop_cpu_cleanup_enable and answer_output_token_ids:
                    model_prompt_ids.extend(answer_output_token_ids)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_output_text}]})
            return _return_output()

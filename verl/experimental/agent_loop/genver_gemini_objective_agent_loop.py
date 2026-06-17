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

import asyncio
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.genver_agent_loop import GenVerAgentLoop
from training.gemini_objectives import (
    build_gemini_runtime_config,
    generate_gemini_answer_judgment,
    generate_gemini_logic_teacher_judgment,
    load_gemini_prompt_template,
)

LOGIC_STEP_EDIT_PATTERN = re.compile(r"(?i)^EDIT_STEP\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
REASON_BLOCK_PATTERN = re.compile(r"(?is)<reason>\s*(?P<body>.*?)\s*</reason>")
REASON_STEP_LINE_PATTERN = re.compile(r"^\s*(?P<idx>\d+)\.\s*(?P<body>.+?)\s*$")
STRICT_REASON_ANSWER_BLOCK_PATTERN = re.compile(
    r"(?is)<(?P<tag>reason|answer)>\s*(?P<payload>.*?)\s*</(?P=tag)>"
)


@register("genver_gemini_objective_agent")
class GenVerGeminiObjectiveAgentLoop(GenVerAgentLoop):
    def __init__(
        self,
        *args: Any,
        loc_verifier_rounds: int = 1,
        logic_verifier_rounds: int = 1,
        logic_verifier_max_new_tokens: int = 96,
        logic_self_verifier_prompt_path: Optional[str] = None,
        gemini_logic_teacher_prompt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, loc_verifier_rounds=loc_verifier_rounds, **kwargs)
        self.loc_verifier_rounds = max(0, int(loc_verifier_rounds))
        del logic_verifier_rounds
        self.logic_verifier_rounds = 1
        self.logic_verifier_max_new_tokens = int(logic_verifier_max_new_tokens)

        if logic_self_verifier_prompt_path:
            prompt_path = Path(logic_self_verifier_prompt_path)
        else:
            prompt_path = (
                self._resolve_training_prompts_dir()
                / "logic_self_verifier_gemini_instructions.txt"
            )
        self.logic_self_verifier_template = self._read_prompt_file(prompt_path)
        self.gemini_logic_teacher_prompt = load_gemini_prompt_template(
            gemini_logic_teacher_prompt_path,
            default_filename="gemini_logic_teacher_judge_instructions.txt",
        )
        reward_cfg = OmegaConf.to_container(
            self.config.get("custom_reward_function", {}),
            resolve=True,
        )
        self.gemini_runtime_config = build_gemini_runtime_config(
            reward_cfg if isinstance(reward_cfg, dict) else {}
        )
        self.gemini_call_semaphore = asyncio.Semaphore(
            max(1, int(self.gemini_runtime_config.get("max_workers", 1)))
        )
        self.gemini_answer_grader_prompt = load_gemini_prompt_template(
            str(
                self.config.get("custom_reward_function", {}).get(
                    "gemini_answer_grader_prompt_path", ""
                )
                or ""
            ).strip()
            or None,
            default_filename="gemini_answer_grader_instructions.txt",
        )
        self.total_training_steps = int(
            OmegaConf.select(
                self.config,
                "actor_rollout_ref.actor.optim.total_training_steps",
                default=0,
            )
            or 0
        )

    def _resolve_training_prompts_dir(self) -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "training" / "prompts").exists():
                return parent / "training" / "prompts"
        raise FileNotFoundError(
            "Could not locate training/prompts relative to the repo."
        )

    def _build_logic_self_verifier_prompt(
        self, query: str, latest_bbox_block: str, latest_answer_output: str
    ) -> str:
        return self.logic_self_verifier_template.format(
            query=query.strip(),
            detected_objects=str(latest_bbox_block or "").strip(),
            answer=str(latest_answer_output or "").strip(),
        )

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str) -> str:
        query_text = query.strip()
        return (
            f"Original query:\n{query_text}\n\n"
            "Detected objects (in [x_min, y_min, x_max, y_max] format with coordinates in [0,1000]):\n"
            f"{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags using numbered steps "
            "(1., 2., 3., ... one step per line) and then putting ONLY your final "
            "answer inside <answer>. Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    def _extract_reason_step_indices(self, answer_output: str) -> list[int]:
        cleaned = str(answer_output or "").replace("<|im_end|>", "").strip()
        match = REASON_BLOCK_PATTERN.search(cleaned)
        if match is None:
            return []
        step_indices: list[int] = []
        for raw_line in match.group("body").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            step_match = REASON_STEP_LINE_PATTERN.fullmatch(line)
            if step_match is None:
                continue
            try:
                step_idx = int(step_match.group("idx"))
            except (TypeError, ValueError):
                continue
            if step_idx not in step_indices:
                step_indices.append(step_idx)
        return step_indices

    def _is_strict_reason_answer_output(self, text: str) -> bool:
        cleaned = str(text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if not cleaned:
            return False
        matches = list(STRICT_REASON_ANSWER_BLOCK_PATTERN.finditer(cleaned))
        if len(matches) == 0:
            return False
        tags: list[str] = []
        cursor = 0
        for match in matches:
            if cleaned[cursor : match.start()].strip():
                return False
            tag = str(match.group("tag") or "").strip().lower()
            payload = str(match.group("payload") or "").strip()
            if not tag or not payload:
                return False
            tags.append(tag)
            cursor = match.end()
        if cleaned[cursor:].strip():
            return False
        return tags == ["reason", "answer"]

    def _parse_logic_step_edits_optional(
        self, text: str, current_answer_output: str
    ) -> tuple[str, bool, dict[str, Any]]:
        cleaned = str(text or "").replace("<|im_end|>", "").strip()
        normalized_lines: list[tuple[str, int]] = []
        line_order = 0
        saw_nonempty_line = False
        valid_step_indices = set(self._extract_reason_step_indices(current_answer_output))
        edited_step_indices: list[int] = []
        seen: set[str] = set()

        for raw_line in cleaned.splitlines():
            if len(normalized_lines) >= 2:
                break
            line = raw_line.strip()
            if not line:
                continue
            saw_nonempty_line = True
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue
            if line.lower().startswith("feedback:"):
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue

            reason_match = LOGIC_STEP_EDIT_PATTERN.match(line)
            if reason_match is not None:
                try:
                    step_idx = int(reason_match.group("idx"))
                except (TypeError, ValueError):
                    continue
                if valid_step_indices and step_idx not in valid_step_indices:
                    continue
                body = reason_match.group("body").strip()
                if body:
                    normalized = f"EDIT_STEP {step_idx}: {body}"
                    if normalized not in seen:
                        normalized_lines.append((normalized, line_order))
                        seen.add(normalized)
                        if step_idx not in edited_step_indices:
                            edited_step_indices.append(step_idx)
                        line_order += 1
                continue

        normalized_lines.sort(key=lambda item: item[1])
        has_effect = len(normalized_lines) > 0
        parse_valid = bool(has_effect or not saw_nonempty_line)
        feedback_info: dict[str, Any] = {
            "logic_feedback_has_effect": bool(has_effect),
            "logic_feedback_valid_for_reward": bool(parse_valid),
            "logic_feedback_has_reason_edit": bool(has_effect),
            "logic_feedback_num_step_edits": int(len(edited_step_indices)),
            "logic_feedback_step_indices": list(edited_step_indices),
        }
        if not has_effect:
            return "", parse_valid, feedback_info
        return "\n".join(line for line, _ in normalized_lines), parse_valid, feedback_info

    def _choose_logic_edit_source(
        self,
        *,
        uid: str,
        global_steps: int,
        validate_mode: bool,
    ) -> str:
        if validate_mode:
            return "self"
        total_steps = max(1, int(self.total_training_steps))
        progress = max(0.0, min(1.0, float(global_steps) / float(total_steps)))
        if progress < 0.2:
            p_self = 0.1
        elif progress < 0.4:
            p_self = 0.3
        elif progress < 0.6:
            p_self = 0.5
        elif progress < 0.8:
            p_self = 0.7
        else:
            p_self = 0.8
        digest = hashlib.sha1(f"{uid}:{global_steps}".encode("utf-8")).digest()
        threshold = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
        return "self" if threshold < p_self else "teacher"

    async def _request_gemini_logic_teacher_judgment(
        self,
        *,
        query: str,
        latest_bbox_block: str,
        current_answer_output: str,
        proposed_self_edits: str,
        images: list[Image.Image],
    ) -> dict[str, Any]:
        async with self.gemini_call_semaphore:
            return await asyncio.to_thread(
                generate_gemini_logic_teacher_judgment,
                config=self.gemini_runtime_config,
                prompt_template=self.gemini_logic_teacher_prompt,
                query=query,
                detected_objects=latest_bbox_block,
                current_answer=current_answer_output,
                proposed_self_edits=proposed_self_edits,
                images=images,
            )

    async def _request_gemini_answer_score(
        self,
        *,
        query: str,
        candidate_response: str,
        images: list[Image.Image],
    ) -> dict[str, Any]:
        async with self.gemini_call_semaphore:
            return await asyncio.to_thread(
                generate_gemini_answer_judgment,
                config=self.gemini_runtime_config,
                prompt_template=self.gemini_answer_grader_prompt,
                query=query,
                candidate_response=candidate_response,
                images=images,
            )

    async def _build_logic_verifier_messages(
        self,
        images: list[Image.Image],
        prompt: str,
        generate_logprobs: bool,
        metrics: dict[str, Any],
    ) -> tuple[
        str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, Any]
    ]:
        messages = self._build_messages(prompt, images)
        prompt_ids, logic_multi_modal_inputs = (
            await self._build_prompt_ids_and_mm_inputs(messages, images)
        )

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": bool(generate_logprobs),
            "max_tokens": int(self.logic_verifier_max_new_tokens),
        }
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
        )
        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        output_ids = output.token_ids
        output_log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        if output_log_probs:
            logic_output_log_probs = [float(x) for x in output_log_probs]
        else:
            logic_output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in images]
        return (
            text,
            prompt_ids,
            output_ids,
            logic_output_log_probs,
            serialized_images,
            logic_multi_modal_inputs,
        )

    async def _build_answer_rewrite_aux_messages(
        self,
        images: list[Image.Image],
        query: str,
        latest_bbox_block: str,
        current_answer_output: str,
        logic_feedback: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[
        str,
        str,
        list[int],
        list[int],
        list[float],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        clean_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        rewrite_prompt = (
            f"{clean_prompt}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            "I have some feedback for you to incorporate. "
            "Please update the <reason> using the feedback, revising the referenced numbered reasoning steps only, "
            "then output a final <answer> that follows from the updated reasoning.\n"
            f"Feedback:\n{logic_feedback}\n"
            "You MUST update the reasoning to incorporate the feedback. "
            "You MUST keep the <reason> step-indexed with numbered lines (1., 2., 3., ... one step per line). "
            "You MUST then produce the final answer from that updated reasoning. "
            "You MUST incorporate the feedback and MUST NOT make unrelated changes. "
            "Please output exactly one full <reason> block and then one full <answer> block. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else. "
            "Do not output any <bbox_2d>."
        )
        messages = self._build_messages(rewrite_prompt, images)
        prompt_ids, answer_multi_modal_inputs = (
            await self._build_prompt_ids_and_mm_inputs(messages, images)
        )
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
            metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        if output_log_probs_raw:
            output_log_probs = [float(x) for x in output_log_probs_raw]
        else:
            output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in images]
        return (
            rewrite_prompt,
            output_text,
            prompt_ids,
            output_ids,
            output_log_probs,
            serialized_images,
            answer_multi_modal_inputs,
        )

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        validate_mode = bool(kwargs.get("validate", False))
        uid = str(kwargs.get("uid", "") or "")
        raw_global_steps = kwargs.get("global_steps", -1)
        global_steps = int(raw_global_steps if raw_global_steps is not None else -1)
        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        raw_images_rgb = [img.convert("RGB") for img in raw_images]
        images = [
            self._resize_longest_side(img, self.max_image_side)
            for img in raw_images_rgb
        ]

        prompted_query = self._build_prompted_context(query)
        messages = self._build_messages(prompted_query, images)
        initial_prompt_ids = await self.apply_chat_template(messages, images=images)
        model_prompt_ids = list(initial_prompt_ids)

        train_prompt_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = (
            [] if sampling_params.get("logprobs") else None
        )
        genver_answer_mask: list[int] = []
        genver_loc_mask: list[int] = []
        genver_loc_calls: list[dict[str, Any]] = []
        genver_loc_verifier_calls: list[dict[str, Any]] = []
        genver_answer_calls: list[dict[str, Any]] = []
        genver_answer_logic_verifier_calls: list[dict[str, Any]] = []
        genver_answer_aux_call: Optional[dict[str, Any]] = None
        loc_call_counter = 0

        output_chunks: list[str] = []
        max_total_tokens = self.response_length
        total_generated_tokens = 0
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'

        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "num_preempted": -1,
        }
        user_turns = 1
        assistant_turns = 0

        def _extend_objective_masks(
            answer_value: int, loc_value: int, count: int
        ) -> None:
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

        def _append_assistant_turn(
            chunk: str, token_ids: list[int], log_probs: list[float]
        ) -> None:
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
                response_logprobs=(
                    response_logprobs[: self.response_length]
                    if response_logprobs is not None
                    else None
                ),
                multi_modal_data={"images": images} if images else {},
                num_turns=user_turns + assistant_turns,
                metrics=metrics,
                extra_fields={
                    "genver_answer_mask": genver_answer_mask[: self.response_length],
                    "genver_loc_mask": genver_loc_mask[: self.response_length],
                    "genver_loc_calls": clipped_loc_calls,
                    "genver_loc_verifier_calls": genver_loc_verifier_calls,
                    "genver_answer_aux_call": genver_answer_aux_call,
                    "genver_answer_calls": genver_answer_calls,
                    "genver_answer_logic_verifier_calls": genver_answer_logic_verifier_calls,
                },
            )

        current_entries: list[Any] = []
        while True:
            if len(response_mask) >= self.response_length:
                break

            if self.agent_loop_cpu_cleanup_enable:
                chunk, token_ids, log_probs = (
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
                chunk, token_ids, log_probs = await self._generate_once(
                    messages,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
            if not token_ids and not chunk.strip():
                break

            total_generated_tokens += len(token_ids)
            assistant_turns += 1
            _append_assistant_turn(chunk, token_ids, log_probs)
            if self.agent_loop_cpu_cleanup_enable and token_ids:
                model_prompt_ids.extend(token_ids)
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": chunk}]}
            )

            if total_generated_tokens >= max_total_tokens:
                break
            if "</answer>" in chunk:
                break

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
                entries_by_image: dict[int, list[Any]] = {}
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

                verifier_prompt = self._build_verifier_prompt(
                    current_entries, len(images)
                )
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
                corrections, _, feedback_info = self._parse_verifier_feedback(
                    verifier_output, current_entries
                )
                current_loc_call_index = (
                    genver_loc_calls[-1]["call_index"] if genver_loc_calls else -1
                )
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
                                feedback_info.get(
                                    "feedback_has_duplicate_add_existing", False
                                )
                            ),
                            "genver_loc_verifier_feedback_has_disallowed_remove": bool(
                                feedback_info.get(
                                    "feedback_has_disallowed_remove", False
                                )
                            ),
                            "genver_loc_verifier_feedback_has_invalid_remove": bool(
                                feedback_info.get("feedback_has_invalid_remove", False)
                            ),
                            "genver_loc_verifier_feedback_has_remove_add_duplicate": bool(
                                feedback_info.get(
                                    "feedback_has_remove_add_duplicate", False
                                )
                            ),
                            "genver_loc_verifier_feedback_duplicate_add_existing_count": int(
                                feedback_info.get(
                                    "feedback_duplicate_add_existing_count", 0
                                )
                            ),
                            "genver_loc_verifier_feedback_disallowed_remove_count": int(
                                feedback_info.get("feedback_disallowed_remove_count", 0)
                            ),
                            "genver_loc_verifier_feedback_invalid_remove_count": int(
                                feedback_info.get("feedback_invalid_remove_count", 0)
                            ),
                            "genver_loc_verifier_feedback_remove_add_duplicate_count": int(
                                feedback_info.get(
                                    "feedback_remove_add_duplicate_count", 0
                                )
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
                    correction_chunk, correction_token_ids, correction_log_probs = (
                        await self._generate_once(
                            messages,
                            images=images,
                            sampling_params=sampling_params,
                            stop_sequences=["</bbox_2d>"],
                            max_new_tokens=self.max_new_tokens_per_chunk,
                            metrics=metrics,
                        )
                    )
                total_generated_tokens += len(correction_token_ids)
                assistant_turns += 1
                _append_assistant_turn(
                    correction_chunk, correction_token_ids, correction_log_probs
                )
                if self.agent_loop_cpu_cleanup_enable and correction_token_ids:
                    model_prompt_ids.extend(correction_token_ids)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": correction_chunk}],
                    }
                )

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

            # Grounding stage for this variant is a single initial bbox call
            # followed by verifier refinements, then answer compaction.
            break

        # Best-effort recovery: if grounding parsing failed to update current_entries,
        # use the last parsable bbox block from generated text so answer compaction
        # still has a concrete detection context.
        if len(current_entries) == 0:
            fallback_payloads = self._extract_bbox_2d_payloads("".join(output_chunks))
            if len(fallback_payloads) > 0:
                fallback_entries = self._parse_bbox_2d_entries(fallback_payloads[-1])
                if fallback_entries and not self._has_invalid_box(fallback_entries):
                    current_entries = fallback_entries

        latest_bbox_block = self._format_bbox_block(current_entries)
        gemini_images = raw_images_rgb if raw_images_rgb else images

        # Initial compacted answer call.
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

        current_answer_output = str(answer_output_text or "")
        answer_call_index = 0
        current_answer_aux_record: dict[str, Any] = {
            "call_index": int(answer_call_index),
            "answer_prompt_text": answer_prompt_text,
            "answer_output_text": current_answer_output,
            "answer_prompt_token_ids": answer_prompt_token_ids,
            "answer_output_token_ids": answer_output_token_ids,
            "answer_output_log_probs": answer_output_log_probs,
            "answer_images": answer_images,
            "answer_multi_modal_inputs": answer_multi_modal_inputs,
            "answer_latest_bbox_block": latest_bbox_block,
            "answer_solution_str": f"{latest_bbox_block}\n{current_answer_output}",
        }
        genver_answer_calls.append(
            {
                "call_index": int(answer_call_index),
                "answer_prompt_text": answer_prompt_text,
                "answer_output_text": current_answer_output,
                "answer_solution_str": current_answer_aux_record[
                    "answer_solution_str"
                ],
            }
        )

        logic_prompt_text = self._build_logic_self_verifier_prompt(
            query, latest_bbox_block, current_answer_output
        )
        (
            logic_output_text,
            logic_prompt_token_ids,
            logic_output_token_ids,
            logic_output_log_probs,
            logic_images,
            logic_multi_modal_inputs,
        ) = await self._build_logic_verifier_messages(
            images=images,
            prompt=logic_prompt_text,
            generate_logprobs=bool(sampling_params.get("logprobs", False)),
            metrics=metrics,
        )
        self_feedback, self_parse_valid, self_feedback_info = (
            self._parse_logic_step_edits_optional(
                logic_output_text, current_answer_output
            )
        )

        if validate_mode:
            teacher_judgment: dict[str, Any] = {}
            logic_teacher_time_s = 0.0
            teacher_output_text = ""
            teacher_feedback = ""
            teacher_parse_valid = True
            teacher_feedback_info = {
                "logic_feedback_has_effect": False,
                "logic_feedback_valid_for_reward": True,
                "logic_feedback_has_reason_edit": False,
                "logic_feedback_num_step_edits": 0,
                "logic_feedback_step_indices": [],
            }
            current_answer_score = 0.0
            self_edit_score = 0.0
            self_edit_reason = ""
        else:
            t_logic_teacher_start = time.perf_counter()
            teacher_judgment = await self._request_gemini_logic_teacher_judgment(
                query=query,
                latest_bbox_block=latest_bbox_block,
                current_answer_output=current_answer_output,
                proposed_self_edits=logic_output_text,
                images=gemini_images,
            )
            logic_teacher_time_s = float(time.perf_counter() - t_logic_teacher_start)
            teacher_edits_raw = teacher_judgment.get("teacher_edits", [])
            if isinstance(teacher_edits_raw, (list, tuple)):
                teacher_output_text = "\n".join(
                    str(line or "").strip()
                    for line in teacher_edits_raw
                    if str(line or "").strip()
                )
            else:
                teacher_output_text = str(teacher_edits_raw or "").strip()
            teacher_feedback, teacher_parse_valid, teacher_feedback_info = (
                self._parse_logic_step_edits_optional(
                    teacher_output_text, current_answer_output
                )
            )
            current_answer_score = float(
                teacher_judgment.get("current_answer_score", 0.0) or 0.0
            )
            self_edit_score = float(
                teacher_judgment.get("self_edit_score", 0.0) or 0.0
            )
            self_edit_reason = str(
                teacher_judgment.get("self_edit_reason", "") or ""
            ).strip()
        teacher_judgment_failed = bool(teacher_judgment.get("failed", False))

        edit_source = self._choose_logic_edit_source(
            uid=uid,
            global_steps=global_steps,
            validate_mode=validate_mode,
        )
        force_logic_reward_invalid = False
        if not validate_mode and teacher_judgment_failed:
            force_logic_reward_invalid = True
            if edit_source == "teacher":
                edit_source = "self"

        if edit_source == "self":
            selected_feedback = self_feedback
            selected_feedback_info = self_feedback_info
        else:
            selected_feedback = teacher_feedback
            selected_feedback_info = teacher_feedback_info
        if validate_mode and not str(selected_feedback or "").strip():
            selected_feedback = (
                "No valid self-verifier feedback was produced. "
                "Re-emit the current answer unchanged."
            )

        logic_call_record = {
            "round_index": 0,
            "answer_call_index": int(answer_call_index),
            "logic_feedback": str(self_feedback),
            "logic_feedback_parse_valid": bool(self_parse_valid),
            "logic_feedback_valid_for_reward": bool(
                self_feedback_info.get("logic_feedback_valid_for_reward", False)
            )
            and not force_logic_reward_invalid,
            "logic_feedback_has_reason_edit": bool(
                self_feedback_info.get("logic_feedback_has_reason_edit", False)
            ),
            "logic_feedback_num_step_edits": int(
                self_feedback_info.get("logic_feedback_num_step_edits", 0)
            ),
            "logic_feedback_step_indices": list(
                self_feedback_info.get("logic_feedback_step_indices", [])
            ),
            "logic_verifier_prompt_text": logic_prompt_text,
            "logic_verifier_output_text": logic_output_text,
            "logic_verifier_prompt_token_ids": logic_prompt_token_ids,
            "logic_verifier_output_token_ids": logic_output_token_ids,
            "logic_verifier_output_log_probs": logic_output_log_probs,
            "logic_verifier_images": logic_images,
            "logic_verifier_multi_modal_inputs": logic_multi_modal_inputs,
            "logic_teacher_output_text": teacher_output_text,
            "logic_teacher_output_raw_text": str(
                teacher_judgment.get("raw_text", "") or ""
            ),
            "logic_teacher_time_s": float(logic_teacher_time_s),
            "logic_teacher_skipped_validate": bool(validate_mode),
            "logic_teacher_feedback": str(teacher_feedback),
            "logic_teacher_parse_valid": bool(teacher_parse_valid),
            "logic_teacher_num_step_edits": int(
                teacher_feedback_info.get("logic_feedback_num_step_edits", 0)
            ),
            "logic_teacher_step_indices": list(
                teacher_feedback_info.get("logic_feedback_step_indices", [])
            ),
            "logic_edit_source": str(edit_source),
            "logic_selected_feedback": str(selected_feedback),
            "logic_selected_num_step_edits": int(
                selected_feedback_info.get("logic_feedback_num_step_edits", 0)
            ),
            "genver_answer_logic_verifier_self_edit_score": float(self_edit_score),
            "genver_answer_logic_verifier_self_edit_reason": self_edit_reason,
            "genver_answer_logic_verifier_current_answer_score": float(current_answer_score),
            "genver_answer_logic_verifier_final_answer_score": float(current_answer_score),
            "genver_answer_logic_verifier_rewrite_skipped_no_edits": False,
        }
        genver_answer_logic_verifier_calls.append(logic_call_record)

        final_answer_call = dict(current_answer_aux_record)
        final_answer_score = float(current_answer_score)
        rewrite_skipped_no_edits = not bool(selected_feedback.strip())
        if rewrite_skipped_no_edits:
            # Validation path: do not force zero override; let validation reward compute normally.
            if validate_mode:
                final_answer_score = 0.0
                final_answer_call["answer_gemini_score_time_s"] = 0.0
            # Training path: if teacher judgment is valid, reuse current score.
            elif not teacher_judgment_failed:
                final_answer_score = float(current_answer_score)
                final_answer_call["answer_reward_override"] = float(current_answer_score)
                final_answer_call["answer_gemini_score_time_s"] = 0.0
            # Training fallback: teacher judgment failed, so grade the current answer directly.
            else:
                t_answer_grade_start = time.perf_counter()
                final_answer_judgment = await self._request_gemini_answer_score(
                    query=query,
                    candidate_response=final_answer_call["answer_solution_str"],
                    images=gemini_images,
                )
                answer_gemini_score_time_s = float(
                    time.perf_counter() - t_answer_grade_start
                )
                final_answer_score = float(
                    final_answer_judgment.get("score", 0.0) or 0.0
                )
                final_answer_call["answer_reward_override"] = float(final_answer_score)
                final_answer_call["answer_gemini_score_time_s"] = float(
                    answer_gemini_score_time_s
                )
        else:
            (
                rewrite_prompt_text,
                rewrite_output_text,
                rewrite_prompt_token_ids,
                rewrite_output_token_ids,
                rewrite_output_log_probs,
                rewrite_images,
                rewrite_multi_modal_inputs,
            ) = await self._build_answer_rewrite_aux_messages(
                images=images,
                query=query,
                latest_bbox_block=latest_bbox_block,
                current_answer_output=current_answer_output,
                logic_feedback=selected_feedback,
                sampling_params=sampling_params,
                metrics=metrics,
            )

            answer_call_index += 1
            current_answer_output = str(rewrite_output_text or "")
            final_answer_call = {
                "call_index": int(answer_call_index),
                "answer_prompt_text": rewrite_prompt_text,
                "answer_output_text": current_answer_output,
                "answer_prompt_token_ids": rewrite_prompt_token_ids,
                "answer_output_token_ids": rewrite_output_token_ids,
                "answer_output_log_probs": rewrite_output_log_probs,
                "answer_images": rewrite_images,
                "answer_multi_modal_inputs": rewrite_multi_modal_inputs,
                "answer_latest_bbox_block": latest_bbox_block,
                "answer_solution_str": f"{latest_bbox_block}\n{current_answer_output}",
            }
            if validate_mode:
                final_answer_score = 0.0
                final_answer_call["answer_gemini_score_time_s"] = 0.0
            else:
                t_answer_grade_start = time.perf_counter()
                final_answer_judgment = await self._request_gemini_answer_score(
                    query=query,
                    candidate_response=final_answer_call["answer_solution_str"],
                    images=gemini_images,
                )
                answer_gemini_score_time_s = float(
                    time.perf_counter() - t_answer_grade_start
                )
                final_answer_score = float(
                    final_answer_judgment.get("score", 0.0) or 0.0
                )
                final_answer_call["answer_reward_override"] = float(final_answer_score)
                final_answer_call["answer_gemini_score_time_s"] = float(
                    answer_gemini_score_time_s
                )
            genver_answer_calls.append(
                {
                    "call_index": int(answer_call_index),
                    "answer_prompt_text": rewrite_prompt_text,
                    "answer_output_text": current_answer_output,
                    "answer_solution_str": final_answer_call["answer_solution_str"],
                }
            )

        answer_format_valid = self._is_strict_reason_answer_output(
            str(final_answer_call.get("answer_output_text", "") or "")
        )
        final_answer_call["answer_format_valid_for_reward"] = bool(answer_format_valid)
        if not answer_format_valid:
            final_answer_score = 0.0
            final_answer_call["answer_reward_override"] = 0.0

        if not validate_mode:
            final_answer_call["answer_reward_override"] = float(final_answer_score)
        if "answer_gemini_score_time_s" not in final_answer_call:
            final_answer_call["answer_gemini_score_time_s"] = 0.0
        final_answer_call["gemini_total_time_s"] = float(
            logic_teacher_time_s
            + float(final_answer_call.get("answer_gemini_score_time_s", 0.0) or 0.0)
        )
        logic_call_record["genver_answer_logic_verifier_final_answer_score"] = float(
            final_answer_score
        )
        logic_call_record["genver_answer_logic_verifier_rewrite_skipped_no_edits"] = bool(
            rewrite_skipped_no_edits
        )
        logic_call_record["genver_answer_logic_verifier_logic_teacher_time_s"] = float(
            logic_teacher_time_s
        )
        logic_call_record["genver_answer_logic_verifier_answer_gemini_score_time_s"] = float(
            final_answer_call.get("answer_gemini_score_time_s", 0.0) or 0.0
        )
        logic_call_record["genver_answer_logic_verifier_gemini_total_time_s"] = float(
            final_answer_call.get("gemini_total_time_s", 0.0) or 0.0
        )

        genver_answer_aux_call = (
            dict(final_answer_call) if isinstance(final_answer_call, dict) else None
        )

        if validate_mode and isinstance(final_answer_call, dict):
            final_output_token_ids = list(
                final_answer_call.get("answer_output_token_ids", []) or []
            )
            final_output_log_probs_raw = list(
                final_answer_call.get("answer_output_log_probs", []) or []
            )
            final_output_log_probs = [
                float(x) if isinstance(x, (int, float)) else 0.0
                for x in final_output_log_probs_raw
            ]
            final_output_text = str(
                final_answer_call.get("answer_output_text", "") or ""
            )
            total_generated_tokens += len(final_output_token_ids)
            assistant_turns += 1
            _append_assistant_turn(
                final_output_text, final_output_token_ids, final_output_log_probs
            )
            if self.agent_loop_cpu_cleanup_enable and final_output_token_ids:
                model_prompt_ids.extend(final_output_token_ids)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_output_text}],
                }
            )
        return _return_output()

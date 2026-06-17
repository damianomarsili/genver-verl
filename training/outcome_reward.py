#!/usr/bin/env python3
"""
Outcome reward for Omni3D-Bench-style answers with answer_type support.

Non-float answers require an exact match between prediction and ground truth.
Float answers accept predictions within a 15% relative error.
"""

from __future__ import annotations

import json
import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment


ANSWER_OPEN_PATTERN = re.compile(r"(?is)<answer>")
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
CUDA_DEVICE_PATTERN = re.compile(r"^cuda(?::\d+)?$")
BBOX_2D_BLOCK_PATTERN = re.compile(r"(?is)<bbox_2d>(.*?)</bbox_2d>")
BBOX_2D_LINE_PATTERN = re.compile(
    r'^\s*(?P<idx>\d+)\s*:\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*'
    r"\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*$",
    re.IGNORECASE,
)
GENVER_TAG_BLOCK_PATTERN = re.compile(
    r"(?is)<(?P<tag>bbox_2d|verifier|reason|answer)>\s*(?P<payload>.*?)\s*</(?P=tag)>"
)


_GENVER_SAM3_RUNTIMES: dict[tuple[str, float, str, bool], dict[str, Any]] = {}
_GENVER_SAM3_RUNTIME_LOCK = threading.Lock()
_GENVER_SAM3_LABEL_CACHE: dict[tuple[Any, ...], list[tuple[float, float, float, float]]] = {}
_GENVER_SAM3_LABEL_CACHE_LOCK = threading.Lock()


def _extract_answer(text: str) -> str | None:
    if not text:
        return None
    open_matches = list(ANSWER_OPEN_PATTERN.finditer(text))
    if not open_matches:
        return None
    match = open_matches[-1]
    content_start = match.end()
    close_match = re.search(r"(?is)</answer>", text[content_start:])
    if close_match:
        content_end = content_start + close_match.start()
        return text[content_start:content_end].strip()
    next_tag = re.search(r"(?is)<(reason|depth|bbox_2d|verifier|answer)>", text[content_start:])
    if next_tag:
        content_end = content_start + next_tag.start()
        return text[content_start:content_end].strip()
    return text[content_start:].strip()


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


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_cuda_device(device: Any) -> str:
    normalized = str(device or "cuda").strip().lower()
    if not CUDA_DEVICE_PATTERN.fullmatch(normalized):
        raise RuntimeError(
            f"GENVER SAM3 reward only supports CUDA devices like 'cuda' or 'cuda:N', got device='{device}'."
        )
    if ":" in normalized:
        requested_index = int(normalized.split(":", 1)[1])
        if requested_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"GENVER SAM3 reward requested {normalized}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return normalized


def _cuda_device_index(device: Any) -> int:
    normalized = _normalize_cuda_device(device)
    if normalized == "cuda":
        return 0
    return int(normalized.split(":", 1)[1])


def _resolve_sam3_worker_devices(
    *,
    shard_enable: bool,
    default_device: str,
    explicit_devices: Any,
    shard_workers: Any,
) -> list[str]:
    base_device = _normalize_cuda_device(default_device)
    if not shard_enable:
        return [base_device]

    devices: list[str] = []
    if isinstance(explicit_devices, str):
        for token in explicit_devices.split(","):
            candidate = token.strip()
            if not candidate:
                continue
            devices.append(_normalize_cuda_device(candidate))
    elif isinstance(explicit_devices, (list, tuple)):
        for candidate in explicit_devices:
            token = str(candidate or "").strip()
            if not token:
                continue
            devices.append(_normalize_cuda_device(token))

    if len(devices) == 0:
        visible = int(torch.cuda.device_count())
        if visible > 1:
            devices = [f"cuda:{idx}" for idx in range(visible)]
        else:
            devices = [base_device]

    deduped: list[str] = []
    for device in devices:
        if device not in deduped:
            deduped.append(device)

    try:
        worker_limit = int(shard_workers)
    except (TypeError, ValueError):
        worker_limit = 0
    if worker_limit > 0:
        deduped = deduped[:worker_limit]
    if len(deduped) == 0:
        deduped = [base_device]
    return deduped


def _normalize_label(label: Any) -> str:
    return " ".join(str(label or "").strip().lower().split())


def _normalize_for_query_match(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ""
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _tokens_for_query_match(text: Any) -> set[str]:
    normalized = _normalize_for_query_match(text)
    if not normalized:
        return set()
    tokens = normalized.split()
    expanded: set[str] = set()
    for token in tokens:
        if not token:
            continue
        expanded.add(token)
        if len(token) > 3 and token.endswith("ies"):
            expanded.add(token[:-3] + "y")
        if len(token) > 3 and token.endswith("es"):
            expanded.add(token[:-2])
        if len(token) > 2 and token.endswith("s"):
            expanded.add(token[:-1])
    return expanded


def _label_is_in_query(label: str, query_tokens: set[str]) -> bool:
    if len(query_tokens) == 0:
        return False
    label_tokens = _tokens_for_query_match(label)
    if len(label_tokens) == 0:
        return False
    return all(token in query_tokens for token in label_tokens)


def _build_sam3_label_candidates(label: str) -> list[str]:
    base = _normalize_label(label)
    if not base:
        return []
    variants = [
        base,
        base.replace("_", " "),
        base.replace("-", " "),
        base.replace("_", " ").replace("-", " "),
    ]
    deduped: list[str] = []
    for variant in variants:
        normalized = _normalize_label(variant)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _box_1000_to_xyxy_pixels(
    box_1000: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    def _scale_coord_1000(x: float, y: float) -> tuple[int, int]:
        x_px = int(round(float(x) / 1000.0 * width))
        y_px = int(round(float(y) / 1000.0 * height))
        x_px = max(0, min(width - 1, x_px))
        y_px = max(0, min(height - 1, y_px))
        return x_px, y_px

    x1, y1, x2, y2 = box_1000
    left_top = _scale_coord_1000(x1, y1)
    right_bottom = _scale_coord_1000(x2, y2)
    left = float(min(left_top[0], right_bottom[0]))
    top = float(min(left_top[1], right_bottom[1]))
    right = float(max(left_top[0], right_bottom[0]))
    bottom = float(max(left_top[1], right_bottom[1]))
    return (left, top, right, bottom)


def _compute_iou_matrix_xyxy(
    loc_boxes_xyxy: list[tuple[float, float, float, float]],
    sam_boxes_xyxy: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Vectorized pairwise IoU matrix (loc x sam)."""
    if len(loc_boxes_xyxy) == 0 or len(sam_boxes_xyxy) == 0:
        return np.zeros((len(loc_boxes_xyxy), len(sam_boxes_xyxy)), dtype=np.float32)

    loc = np.asarray(loc_boxes_xyxy, dtype=np.float32)
    sam = np.asarray(sam_boxes_xyxy, dtype=np.float32)

    inter_x1 = np.maximum(loc[:, None, 0], sam[None, :, 0])
    inter_y1 = np.maximum(loc[:, None, 1], sam[None, :, 1])
    inter_x2 = np.minimum(loc[:, None, 2], sam[None, :, 2])
    inter_y2 = np.minimum(loc[:, None, 3], sam[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    loc_area = np.maximum(0.0, loc[:, 2] - loc[:, 0]) * np.maximum(0.0, loc[:, 3] - loc[:, 1])
    sam_area = np.maximum(0.0, sam[:, 2] - sam[:, 0]) * np.maximum(0.0, sam[:, 3] - sam[:, 1])
    union = loc_area[:, None] + sam_area[None, :] - inter_area

    iou = np.zeros_like(inter_area, dtype=np.float32)
    valid = union > 0.0
    iou[valid] = inter_area[valid] / union[valid]
    return iou


def _pq_match_boxes(
    loc_boxes: list[dict[str, Any]],
    sam_boxes: list[dict[str, Any]],
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Match predicted and SAM boxes with one-to-one max-IoU assignment.

    Returns:
        matches: (loc_idx, sam_idx, iou) from global one-to-one matching.
        iou_matrix: IoU matrix over all loc/sam pairs.
    """
    num_loc = len(loc_boxes)
    num_sam = len(sam_boxes)
    if num_loc == 0 or num_sam == 0:
        return [], np.zeros((num_loc, num_sam), dtype=np.float32)

    iou_matrix = _compute_iou_matrix_xyxy(
        [tuple(loc["box_xyxy"]) for loc in loc_boxes],
        [tuple(sam["box_xyxy"]) for sam in sam_boxes],
    )

    # Continuous matching: maximize total IoU directly with no hard IoU gate.
    cost_matrix = -iou_matrix.astype(np.float32, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int, float]] = []
    for loc_idx, sam_idx in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
        if loc_idx >= num_loc or sam_idx >= num_sam:
            continue
        iou = float(iou_matrix[loc_idx, sam_idx])
        matches.append((loc_idx, sam_idx, iou))
    return matches, iou_matrix


def _extract_bbox_2d_payloads(text: str) -> list[str]:
    if not text:
        return []
    payloads: list[str] = []
    for raw_payload in BBOX_2D_BLOCK_PATTERN.findall(text):
        payloads.append(str(raw_payload).strip())
    return payloads


def _parse_bbox_2d_entries_from_text(bbox_2d_text: str) -> list[dict[str, Any]]:
    payloads = _extract_bbox_2d_payloads(bbox_2d_text)
    # Strict format: exactly one <bbox_2d> block with one line per object.
    if len(payloads) != 1:
        return []
    return _parse_bbox_2d_entries_from_payload(payloads[0])


def _parse_bbox_2d_entries_from_payload(payload: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    nonempty_line_count = 0
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        nonempty_line_count += 1
        match = BBOX_2D_LINE_PATTERN.fullmatch(line)
        if match is None:
            return []
        try:
            idx = int(match.group("idx"))
        except (TypeError, ValueError):
            return []
        if idx != len(entries) + 1:
            return []
        label = _normalize_label(match.group("label"))
        if not label:
            return []
        box_1000 = (
            float(match.group("x1")),
            float(match.group("y1")),
            float(match.group("x2")),
            float(match.group("y2")),
        )
        x1, y1, x2, y2 = box_1000
        if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
            return []
        if x2 <= x1 or y2 <= y1:
            return []
        entries.append({"image_index": 1, "label": label, "box_1000": box_1000})
    if nonempty_line_count == 0:
        return []
    return entries


def _extract_bbox_2d_entries_from_call_record(call_record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = call_record.get("parsed_entries_1000")
    if isinstance(raw_entries, np.ndarray):
        raw_entries = raw_entries.tolist()
    if isinstance(raw_entries, (list, tuple)):
        entries: list[dict[str, Any]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                entries = []
                break
            image_index = _safe_int(raw_entry.get("image_index", 1), 1)
            label = _normalize_label(raw_entry.get("label", ""))
            box = raw_entry.get("box_1000")
            if (
                not label
                or not isinstance(box, (list, tuple))
                or len(box) != 4
            ):
                entries = []
                break
            try:
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            except (TypeError, ValueError):
                entries = []
                break
            if not (
                0.0 <= x1 <= 1000.0
                and 0.0 <= y1 <= 1000.0
                and 0.0 <= x2 <= 1000.0
                and 0.0 <= y2 <= 1000.0
            ):
                entries = []
                break
            if x2 <= x1 or y2 <= y1:
                entries = []
                break
            entries.append(
                {
                    "image_index": int(max(1, image_index)),
                    "label": label,
                    "box_1000": (x1, y1, x2, y2),
                }
            )
        if len(entries) > 0:
            return entries
    return _parse_bbox_2d_entries_from_text(str(call_record.get("text", "") or ""))


def _serialize_image_for_hash(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _coerce_raw_prompt_image(image_entry: Any) -> tuple[Image.Image, str] | None:
    image_obj: Image.Image | None = None
    image_bytes: bytes | None = None

    if isinstance(image_entry, Image.Image):
        image_obj = image_entry.convert("RGB")
    elif isinstance(image_entry, dict):
        if isinstance(image_entry.get("image"), Image.Image):
            image_obj = image_entry["image"].convert("RGB")
        elif isinstance(image_entry.get("bytes"), (bytes, bytearray)):
            image_bytes = bytes(image_entry["bytes"])
            try:
                image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception:
                image_obj = None
        elif image_entry.get("path"):
            try:
                image_obj = Image.open(str(image_entry["path"])).convert("RGB")
            except Exception:
                image_obj = None
    elif isinstance(image_entry, (bytes, bytearray)):
        image_bytes = bytes(image_entry)
        try:
            image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            image_obj = None

    if image_obj is None:
        return None
    if image_bytes is None:
        image_bytes = _serialize_image_for_hash(image_obj)
    image_fingerprint = hashlib.sha1(image_bytes).hexdigest()
    return image_obj, image_fingerprint


def _extract_original_images_from_raw_prompt(raw_prompt: Any) -> list[tuple[Image.Image, str]]:
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, (list, tuple)):
        return []

    images: list[tuple[Image.Image, str]] = []
    for message in raw_prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            break
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            coerced = _coerce_raw_prompt_image(item)
            if coerced is not None:
                images.append(coerced)
        break
    return images


def _extract_query_text_from_raw_prompt(raw_prompt: Any) -> str:
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if not isinstance(raw_prompt, (list, tuple)):
        return ""

    for message in raw_prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", "")).strip()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts).strip()
    return ""


def _serialize_entries_for_dedupe(entries: Sequence[dict[str, Any]]) -> str:
    serialized: list[list[Any]] = []
    for entry in entries:
        box = entry.get("box_1000")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        serialized.append(
            [
                int(entry.get("image_index", 1)),
                str(entry.get("label", "")),
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ]
        )
    return json.dumps(serialized, separators=(",", ":"), ensure_ascii=True)


def _evaluate_loc_entries_with_sam3(
    entries: Sequence[dict[str, Any]],
    originals: Sequence[tuple[Image.Image, str]],
    *,
    query_tokens: set[str],
    genver_sam3_confidence_threshold: float = 0.5,
    genver_sam3_device: str = "cuda",
    genver_sam3_devices: Any = None,
    genver_sam3_shard_workers: Any = None,
    genver_sam3_shard_enable: bool = True,
    genver_sam3_checkpoint_path: str = "",
    genver_sam3_load_from_hf: bool = True,
) -> dict[str, Any]:
    confidence_threshold = _coerce_float(genver_sam3_confidence_threshold, 0.5)
    load_from_hf = _coerce_bool(genver_sam3_load_from_hf, True)
    device = _normalize_cuda_device(genver_sam3_device)
    shard_enable = _coerce_bool(genver_sam3_shard_enable, True)
    checkpoint_path = str(genver_sam3_checkpoint_path or "")
    worker_devices = _resolve_sam3_worker_devices(
        shard_enable=shard_enable,
        default_device=device,
        explicit_devices=genver_sam3_devices,
        shard_workers=genver_sam3_shard_workers,
    )

    if not entries:
        return {
            "pq": 0.0,
            "num_sam_boxes": 0,
            "num_loc_boxes": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sam3_unique_tasks": 0,
            "sam3_workers_used": 0,
            "prediction_diagnostics": [],
        }
    if not originals:
        return {
            "pq": 0.0,
            "num_sam_boxes": 0,
            "num_loc_boxes": len(entries),
            "tp": 0,
            "fp": len(entries),
            "fn": 0,
            "sam3_unique_tasks": 0,
            "sam3_workers_used": 0,
            "prediction_diagnostics": [],
        }

    loc_boxes: list[dict[str, Any]] = []
    for entry in entries:
        image_pos = int(entry["image_index"]) - 1
        if image_pos < 0 or image_pos >= len(originals):
            continue
        label = _normalize_label(entry["label"])
        if not label:
            continue
        label_in_query = _label_is_in_query(label, query_tokens)
        image, _ = originals[image_pos]
        loc_boxes.append(
            {
                "image_index": int(entry["image_index"]),
                "label": label,
                "label_in_query": bool(label_in_query),
                "box_xyxy": _box_1000_to_xyxy_pixels(entry["box_1000"], image.width, image.height),
                "box_1000": tuple(float(x) for x in entry["box_1000"]),
            }
        )

    if not loc_boxes:
        return {
            "pq": 0.0,
            "num_sam_boxes": 0,
            "num_loc_boxes": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sam3_unique_tasks": 0,
            "sam3_workers_used": 0,
            "prediction_diagnostics": [],
        }

    loc_by_group: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for loc_box in loc_boxes:
        key = (int(loc_box["image_index"]), str(loc_box["label"]))
        loc_by_group.setdefault(key, []).append(loc_box)

    sam_by_group: dict[tuple[int, str], list[dict[str, Any]]] = {}
    group_to_task_key: dict[tuple[int, str], tuple[str, str]] = {}
    task_payloads: dict[tuple[str, str], tuple[Image.Image, str, str]] = {}
    for image_index, label in loc_by_group.keys():
        group_loc_boxes = loc_by_group.get((image_index, label), [])
        if not any(bool(box.get("label_in_query", False)) for box in group_loc_boxes):
            sam_by_group[(image_index, label)] = []
            continue
        image_pos = image_index - 1
        if image_pos < 0 or image_pos >= len(originals):
            sam_by_group[(image_index, label)] = []
            continue
        image, image_fingerprint = originals[image_pos]
        task_key = (image_fingerprint, label)
        group_to_task_key[(image_index, label)] = task_key
        if task_key not in task_payloads:
            task_payloads[task_key] = (image, label, image_fingerprint)

    sam3_task_results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    sam3_shard_metrics: dict[str, float] = {}
    if len(task_payloads) > 0:
        tasks: list[tuple[tuple[str, str], Image.Image, str, str]] = []
        for task_key, payload in task_payloads.items():
            image, label, image_fingerprint = payload
            tasks.append((task_key, image, label, image_fingerprint))
        sam3_task_results, sam3_shard_metrics = _run_sam3_tasks_sharded(
            tasks=tasks,
            devices=worker_devices,
            confidence_threshold=confidence_threshold,
            checkpoint_path=checkpoint_path,
            load_from_hf=load_from_hf,
        )

    for group_key, task_key in group_to_task_key.items():
        image_index, label = group_key
        boxes_xyxy = sam3_task_results.get(task_key, [])
        sam_by_group[group_key] = [
            {
                "image_index": image_index,
                "label": label,
                "box_xyxy": tuple(float(x) for x in box_xyxy),
            }
            for box_xyxy in boxes_xyxy
        ]

    tp = 0
    fp = 0
    fn = 0
    sum_iou = 0.0
    diagnostics: list[dict[str, Any]] = []
    sam_index = 0

    for group_key, group_loc_boxes in loc_by_group.items():
        group_sam_boxes = sam_by_group.get(group_key, [])
        matches, iou_matrix = _pq_match_boxes(
            group_loc_boxes,
            group_sam_boxes,
        )
        match_by_sam = {sam_idx: (loc_idx, iou) for loc_idx, sam_idx, iou in matches}
        match_by_loc = {loc_idx: (sam_idx, iou) for loc_idx, sam_idx, iou in matches}

        group_tp = len(matches)
        group_fp = max(0, len(group_loc_boxes) - group_tp)
        group_fn = max(0, len(group_sam_boxes) - group_tp)
        group_iou_sum = float(sum(iou for _, _, iou in matches))

        tp += group_tp
        fp += group_fp
        fn += group_fn
        sum_iou += group_iou_sum

        covered_loc_indices: set[int] = set()

        for group_sam_idx, sam_box in enumerate(group_sam_boxes):
            best_iou = 0.0
            best_loc_idx = -1
            if iou_matrix.size > 0 and group_sam_idx < iou_matrix.shape[1]:
                column = iou_matrix[:, group_sam_idx]
                best_iou = float(column.max())
                if column.shape[0] > 0:
                    best_loc_idx = int(column.argmax())
                    covered_loc_indices.add(best_loc_idx)
            matched_pair = match_by_sam.get(group_sam_idx)
            matched_loc_idx = matched_pair[0] if matched_pair is not None else -1
            matched_iou = float(matched_pair[1]) if matched_pair is not None else 0.0
            predicted_loc_box_1000 = (
                tuple(float(x) for x in group_loc_boxes[best_loc_idx]["box_1000"])
                if best_loc_idx >= 0
                else None
            )
            predicted_loc_box_xyxy = (
                tuple(float(x) for x in group_loc_boxes[best_loc_idx]["box_xyxy"])
                if best_loc_idx >= 0
                else None
            )
            diagnostics.append(
                {
                    "sam_index": sam_index,
                    "image_index": int(sam_box["image_index"]),
                    "label": str(sam_box["label"]),
                    "sam3_box_xyxy": tuple(float(x) for x in sam_box["box_xyxy"]),
                    "best_iou": float(best_iou),
                    "matched_iou": matched_iou,
                    "matched": matched_pair is not None,
                    "label_in_query": bool(group_loc_boxes[best_loc_idx].get("label_in_query", False))
                    if best_loc_idx >= 0
                    else True,
                    "predicted_box_1000": predicted_loc_box_1000,
                    "predicted_box_xyxy": predicted_loc_box_xyxy,
                    "matched_loc_box_1000": (
                        tuple(float(x) for x in group_loc_boxes[matched_loc_idx]["box_1000"])
                        if matched_loc_idx >= 0
                        else None
                    ),
                }
            )
            sam_index += 1

        for loc_idx, loc_box in enumerate(group_loc_boxes):
            if loc_idx in covered_loc_indices:
                continue
            best_iou = 0.0
            if iou_matrix.size > 0 and loc_idx < iou_matrix.shape[0]:
                row = iou_matrix[loc_idx, :]
                if row.shape[0] > 0:
                    best_iou = float(row.max())

            matched_pair = match_by_loc.get(loc_idx)
            matched_sam_idx = matched_pair[0] if matched_pair is not None else -1
            matched_iou = float(matched_pair[1]) if matched_pair is not None else 0.0
            sam3_box_xyxy = (
                tuple(float(x) for x in group_sam_boxes[matched_sam_idx]["box_xyxy"])
                if matched_sam_idx >= 0 and matched_sam_idx < len(group_sam_boxes)
                else None
            )
            diagnostics.append(
                {
                    "sam_index": -1,
                    "image_index": int(loc_box["image_index"]),
                    "label": str(loc_box["label"]),
                    "sam3_box_xyxy": sam3_box_xyxy,
                    "best_iou": float(best_iou),
                    "matched_iou": matched_iou,
                    "matched": matched_pair is not None,
                    "label_in_query": bool(loc_box.get("label_in_query", False)),
                    "predicted_box_1000": tuple(float(x) for x in loc_box["box_1000"]),
                    "predicted_box_xyxy": tuple(float(x) for x in loc_box["box_xyxy"]),
                    "matched_loc_box_1000": (
                        tuple(float(x) for x in loc_box["box_1000"]) if matched_pair is not None else None
                    ),
                }
            )

    denom = float(tp) + 0.5 * float(fp) + 0.5 * float(fn)
    pq = float(sum_iou / denom) if denom > 0.0 else 0.0
    return {
        "pq": pq,
        "num_sam_boxes": sum(len(v) for v in sam_by_group.values()),
        "num_loc_boxes": len(loc_boxes),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "sam3_unique_tasks": int(len(task_payloads)),
        "sam3_workers_used": int(min(len(worker_devices), len(task_payloads))),
        "sam3_shard_metrics": dict(sam3_shard_metrics),
        "prediction_diagnostics": diagnostics,
    }


def _get_sam3_runtime(
    *,
    device: str,
    confidence_threshold: float,
    checkpoint_path: str,
    load_from_hf: bool,
) -> dict[str, Any]:
    normalized_device = _normalize_cuda_device(device)
    device_index = _cuda_device_index(normalized_device)
    if not torch.cuda.is_available():
        raise RuntimeError("GENVER SAM3 reward requires CUDA, but torch.cuda.is_available() is False.")

    runtime_key = (
        f"cuda:{device_index}",
        float(confidence_threshold),
        str(checkpoint_path or ""),
        bool(load_from_hf),
    )
    with _GENVER_SAM3_RUNTIME_LOCK:
        cached = _GENVER_SAM3_RUNTIMES.get(runtime_key)
        if cached is not None:
            return cached

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # SAM3 initialization is device-context sensitive in this environment:
        # bind the target GPU explicitly and use device='cuda'.
        with torch.cuda.device(device_index):
            model = build_sam3_image_model(
                device="cuda",
                eval_mode=True,
                checkpoint_path=(checkpoint_path or None),
                load_from_HF=bool(load_from_hf),
            )
            processor = Sam3Processor(
                model=model,
                device="cuda",
                confidence_threshold=float(confidence_threshold),
            )
        runtime = {"model": model, "processor": processor, "device_index": int(device_index)}
        _GENVER_SAM3_RUNTIMES[runtime_key] = runtime
        return runtime


def _run_sam3_for_label(
    image: Image.Image,
    label: str,
    *,
    image_fingerprint: str,
    device: str,
    confidence_threshold: float,
    checkpoint_path: str,
    load_from_hf: bool,
) -> list[tuple[float, float, float, float]]:
    normalized_label = _normalize_label(label)
    if not normalized_label:
        return []
    normalized_device = _normalize_cuda_device(device)
    device_index = _cuda_device_index(normalized_device)
    cache_key = (
        image_fingerprint,
        normalized_label,
        float(confidence_threshold),
        str(checkpoint_path or ""),
        bool(load_from_hf),
    )
    with _GENVER_SAM3_LABEL_CACHE_LOCK:
        cached = _GENVER_SAM3_LABEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    runtime = _get_sam3_runtime(
        device=normalized_device,
        confidence_threshold=confidence_threshold,
        checkpoint_path=checkpoint_path,
        load_from_hf=load_from_hf,
    )
    processor = runtime["processor"]

    boxes: list[tuple[float, float, float, float]] = []
    best_count = -1
    for label_candidate in _build_sam3_label_candidates(normalized_label):
        with torch.cuda.device(device_index):
            with torch.inference_mode():
                state = processor.set_image(image, state={})
                state = processor.set_text_prompt(prompt=label_candidate, state=state)

        candidate_boxes: list[tuple[float, float, float, float]] = []
        raw_boxes = state.get("boxes")
        if raw_boxes is not None:
            try:
                rows = raw_boxes.detach().cpu().tolist()
            except Exception:
                rows = []
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    left = min(x1, x2)
                    top = min(y1, y2)
                    right = max(x1, x2)
                    bottom = max(y1, y2)
                    candidate_boxes.append((left, top, right, bottom))
        del state

        if len(candidate_boxes) > best_count:
            boxes = candidate_boxes
            best_count = len(candidate_boxes)
        if best_count > 0:
            # First candidate with detections is usually good enough.
            break

    with _GENVER_SAM3_LABEL_CACHE_LOCK:
        cached = _GENVER_SAM3_LABEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        _GENVER_SAM3_LABEL_CACHE[cache_key] = boxes
    return boxes


def _run_sam3_tasks_sharded(
    *,
    tasks: Sequence[tuple[tuple[str, str], Image.Image, str, str]],
    devices: Sequence[str],
    confidence_threshold: float,
    checkpoint_path: str,
    load_from_hf: bool,
) -> tuple[dict[tuple[str, str], list[tuple[float, float, float, float]]], dict[str, float]]:
    if len(tasks) == 0:
        return {}, {}
    worker_devices = list(devices)
    if len(worker_devices) == 0:
        worker_devices = ["cuda"]
    worker_devices = [_normalize_cuda_device(device) for device in worker_devices]

    shards: dict[str, list[tuple[tuple[str, str], Image.Image, str, str]]] = {device: [] for device in worker_devices}
    for idx, task in enumerate(tasks):
        device = worker_devices[idx % len(worker_devices)]
        shards[device].append(task)

    nonempty_devices = [device for device in worker_devices if len(shards[device]) > 0]
    if len(nonempty_devices) <= 1:
        only_device = nonempty_devices[0] if nonempty_devices else worker_devices[0]
        task_results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
        start_t = time.perf_counter()
        for task_key, image, label, image_fingerprint in shards[only_device]:
            task_results[task_key] = _run_sam3_for_label(
                image=image,
                label=label,
                image_fingerprint=image_fingerprint,
                device=only_device,
                confidence_threshold=confidence_threshold,
                checkpoint_path=checkpoint_path,
                load_from_hf=load_from_hf,
            )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return task_results, {f"sam3_time_ms_{only_device}": float(elapsed_ms)}

    def _worker(
        device: str,
        shard_tasks: list[tuple[tuple[str, str], Image.Image, str, str]],
    ) -> tuple[str, dict[tuple[str, str], list[tuple[float, float, float, float]]], float]:
        device_index = _cuda_device_index(device)
        torch.cuda.set_device(device_index)
        start_t = time.perf_counter()
        out: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
        for task_key, image, label, image_fingerprint in shard_tasks:
            out[task_key] = _run_sam3_for_label(
                image=image,
                label=label,
                image_fingerprint=image_fingerprint,
                device=device,
                confidence_threshold=confidence_threshold,
                checkpoint_path=checkpoint_path,
                load_from_hf=load_from_hf,
            )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return device, out, float(elapsed_ms)

    task_results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    metrics: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(nonempty_devices)) as executor:
        futures = [executor.submit(_worker, device, shards[device]) for device in nonempty_devices]
        for future in futures:
            device, out, elapsed_ms = future.result()
            task_results.update(out)
            metrics[f"sam3_time_ms_{device}"] = float(elapsed_ms)
    return task_results, metrics


def _normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.strip().lower().split())
    return normalized or None


def _strip_format_special_tokens(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<|im_end|>", "")
    cleaned = cleaned.replace("<|endoftext|>", "")
    return cleaned.strip()


def _is_strict_bbox_reason_answer_format(text: str) -> bool:
    if not text:
        return False
    cleaned = _strip_format_special_tokens(text)
    tag_blocks = list(GENVER_TAG_BLOCK_PATTERN.finditer(cleaned))
    if len(tag_blocks) == 0:
        return False

    tags: list[str] = []
    payloads: list[str] = []
    cursor = 0
    for match in tag_blocks:
        if cleaned[cursor : match.start()].strip():
            return False
        tag = str(match.group("tag") or "").strip().lower()
        payload = str(match.group("payload") or "").strip()
        if not tag or not payload:
            return False
        tags.append(tag)
        payloads.append(payload)
        cursor = match.end()
    if cleaned[cursor:].strip():
        return False

    # Exactly one final reason+answer, with answer last.
    if tags.count("reason") != 1 or tags.count("answer") != 1:
        return False
    reason_idx = tags.index("reason")
    answer_idx = tags.index("answer")
    if reason_idx != len(tags) - 2 or answer_idx != len(tags) - 1:
        return False

    # Prefix before <reason> must be one-or-more bbox rounds only.
    # Canonical model-output path for GENVER agent loop:
    # <bbox_2d> ... </bbox_2d> (repeated) -> <reason> -> <answer>
    prefix_tags = tags[:reason_idx]
    if len(prefix_tags) == 0 or prefix_tags[0] != "bbox_2d":
        return False
    if any(tag != "bbox_2d" for tag in prefix_tags):
        return False

    # Every bbox block must strictly follow one-object-per-line syntax.
    for tag, payload in zip(tags, payloads, strict=True):
        if tag == "bbox_2d" and len(_parse_bbox_2d_entries_from_payload(payload)) == 0:
            return False
    return True


def _is_strict_reason_answer_format(text: str) -> bool:
    if not text:
        return False
    cleaned = _strip_format_special_tokens(text)
    tag_blocks = list(GENVER_TAG_BLOCK_PATTERN.finditer(cleaned))
    if len(tag_blocks) == 0:
        return False

    tags: list[str] = []
    cursor = 0
    for match in tag_blocks:
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


def _build_format_valid_flags(solution_strs: Sequence[str], num_samples: int) -> list[bool]:
    if isinstance(solution_strs, np.ndarray):
        values = solution_strs.tolist()
    elif isinstance(solution_strs, (list, tuple)):
        values = list(solution_strs)
    else:
        values = []
    flags = [False] * num_samples
    for idx in range(min(num_samples, len(values))):
        flags[idx] = _is_strict_bbox_reason_answer_format(str(values[idx] or ""))
    return flags


def _extract_float(text: str) -> float | None:
    if not text:
        return None
    cleaned = text.strip()
    try:
        return float(cleaned)
    except ValueError:
        pass
    match = NUMBER_PATTERN.search(cleaned)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_float_answer(answer_type: str | None) -> bool:
    if answer_type is None:
        return False
    return answer_type.strip().lower() == "float"


def _exact_match(predicted: str | None, ground_truth: str) -> float:
    pred_norm = _normalize_text(predicted)
    gt_norm = _normalize_text(ground_truth)
    if pred_norm is None or gt_norm is None:
        return 0.0
    return 1.0 if pred_norm == gt_norm else 0.0


def _float_match(predicted: str | None, ground_truth: str) -> float:
    pred_value = _extract_float(predicted or "")
    gt_value = _extract_float(ground_truth)
    if pred_value is None or gt_value is None:
        return 0.0
    if gt_value == 0:
        return 1.0 if abs(pred_value) == 0 else 0.0
    rel_err = abs(pred_value - gt_value) / abs(gt_value)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    matches = sum(1 for theta in thresholds if rel_err < 1.0 - theta)
    return matches / len(thresholds)


def compute_score(solution_str: str, ground_truth: str, answer_type: str | None = None) -> float:
    if not _is_strict_bbox_reason_answer_format(solution_str):
        return 0.0
    predicted = _extract_answer(solution_str)
    if _is_float_answer(answer_type):
        return _float_match(predicted, ground_truth)
    return _exact_match(predicted, ground_truth)


def compute_score_answer_only(solution_str: str, ground_truth: str, answer_type: str | None = None) -> float:
    if not _is_strict_reason_answer_format(solution_str):
        return 0.0
    predicted = _extract_answer(solution_str)
    if _is_float_answer(answer_type):
        return _float_match(predicted, ground_truth)
    return _exact_match(predicted, ground_truth)


def compute_score_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
) -> list[float]:
    scores: list[float] = []
    for solution_str, ground_truth, extra_info in zip(
        solution_strs, ground_truths, extra_infos, strict=True
    ):
        answer_type = None
        if isinstance(extra_info, dict):
            answer_type = extra_info.get("answer_type")
        scores.append(compute_score(solution_str, ground_truth, answer_type))
    return scores


def compute_score_answer_only_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
) -> list[float]:
    del data_sources
    scores: list[float] = []
    for solution_str, ground_truth, extra_info in zip(solution_strs, ground_truths, extra_infos, strict=True):
        answer_type = None
        if isinstance(extra_info, dict):
            answer_type = extra_info.get("answer_type")
        scores.append(compute_score_answer_only(solution_str, ground_truth, answer_type))
    return scores


def compute_loc_call_rewards_batched(
    data_sources: Sequence[str],
    loc_call_records: Sequence[Sequence[dict]] | Iterable[Sequence[dict]],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
    uids: Sequence[str] | Iterable[str] | None = None,
    loc_verifier_call_records: Sequence[Sequence[dict]] | Iterable[Sequence[dict]] | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """Detection objective: reward only final call quality.

    Per sample:
        reward(call=max_call_index) = PQ(max_call_index)
        reward(other calls) = 0.0

    No per-call averaging/division is applied.
    """
    del data_sources, ground_truths, extra_infos, loc_verifier_call_records
    sam3_conf_threshold = _coerce_float(kwargs.get("genver_sam3_confidence_threshold", 0.5), 0.5)
    sam3_device = _normalize_cuda_device(kwargs.get("genver_sam3_device", "cuda"))
    sam3_devices = kwargs.get("genver_sam3_devices")
    sam3_shard_workers = kwargs.get("genver_sam3_shard_workers")
    sam3_checkpoint_path = str(kwargs.get("genver_sam3_checkpoint_path", "") or "")
    sam3_load_from_hf = _coerce_bool(kwargs.get("genver_sam3_load_from_hf", True), True)
    sam3_cache_shard_enable = _coerce_bool(
        kwargs.get("genver_sam3_cache_shard_enable", kwargs.get("genver_sam3_shard_enable", True)),
        True,
    )
    sam3_shard_enable = _coerce_bool(kwargs.get("genver_sam3_shard_enable", sam3_cache_shard_enable), sam3_cache_shard_enable)

    loc_call_records_list: list[list[dict]] = []
    for sample_calls in loc_call_records:
        if isinstance(sample_calls, np.ndarray):
            sample_calls = sample_calls.tolist()
        if not isinstance(sample_calls, (list, tuple)):
            loc_call_records_list.append([])
            continue
        loc_call_records_list.append([record for record in sample_calls if isinstance(record, dict)])

    num_samples = len(loc_call_records_list)
    del uids

    raw_prompts = kwargs.get("raw_prompts")
    if isinstance(raw_prompts, np.ndarray):
        raw_prompts = raw_prompts.tolist()
    if raw_prompts is None:
        raw_prompts_list = [None] * num_samples
    elif isinstance(raw_prompts, (list, tuple)):
        raw_prompts_list = list(raw_prompts)
    else:
        raw_prompts_list = [None] * num_samples
    if len(raw_prompts_list) < num_samples:
        raw_prompts_list.extend([None] * (num_samples - len(raw_prompts_list)))
    format_valid_flags = _build_format_valid_flags(solution_strs, num_samples)
    rewards: list[list[float]] = []
    for sample_idx, call_records in enumerate(loc_call_records_list):
        if not format_valid_flags[sample_idx]:
            sample_rewards = [0.0] * len(call_records)
            for call_record in call_records:
                call_record["genver_loc_reward_quality"] = 0.0
                call_record["genver_loc_reward_raw"] = 0.0
                call_record["genver_loc_reward"] = 0.0
            rewards.append(sample_rewards)
            continue

        raw_prompt = raw_prompts_list[sample_idx] if sample_idx < len(raw_prompts_list) else None
        originals = _extract_original_images_from_raw_prompt(raw_prompt)
        query_tokens = _tokens_for_query_match(_extract_query_text_from_raw_prompt(raw_prompt))
        eval_cache: dict[str, dict[str, Any]] = {}
        pq_values: list[float] = []
        for local_idx, call_record in enumerate(call_records):
            del local_idx
            entries = _extract_bbox_2d_entries_from_call_record(call_record)
            if sam3_cache_shard_enable:
                dedupe_key = _serialize_entries_for_dedupe(entries)
                cached_eval = eval_cache.get(dedupe_key)
                if cached_eval is None:
                    cached_eval = _evaluate_loc_entries_with_sam3(
                        entries=entries,
                        originals=originals,
                        query_tokens=query_tokens,
                        genver_sam3_confidence_threshold=sam3_conf_threshold,
                        genver_sam3_device=sam3_device,
                        genver_sam3_devices=sam3_devices,
                        genver_sam3_shard_workers=sam3_shard_workers,
                        genver_sam3_shard_enable=sam3_shard_enable,
                        genver_sam3_checkpoint_path=sam3_checkpoint_path,
                        genver_sam3_load_from_hf=sam3_load_from_hf,
                    )
                    eval_cache[dedupe_key] = cached_eval
                eval_result = dict(cached_eval)
            else:
                eval_result = _evaluate_loc_entries_with_sam3(
                    entries=entries,
                    originals=originals,
                    query_tokens=query_tokens,
                    genver_sam3_confidence_threshold=sam3_conf_threshold,
                    genver_sam3_device=sam3_device,
                    genver_sam3_devices=sam3_devices,
                    genver_sam3_shard_workers=sam3_shard_workers,
                    genver_sam3_shard_enable=False,
                    genver_sam3_checkpoint_path=sam3_checkpoint_path,
                    genver_sam3_load_from_hf=sam3_load_from_hf,
                )
            call_record["genver_sam3_eval"] = dict(eval_result)
            pq_values.append(_clip01(eval_result.get("pq", 0.0)))

        call_indices: list[int] = []
        for idx in range(len(call_records)):
            quality = _clip01(pq_values[idx] if idx < len(pq_values) else 0.0)
            call_index = _safe_int(call_records[idx].get("call_index", idx), idx)
            call_indices.append(call_index)
            call_records[idx]["genver_loc_reward_quality"] = float(quality)
            call_records[idx]["genver_loc_reward_raw"] = 0.0

        sample_rewards = [0.0] * len(call_records)
        if len(call_records) > 0:
            last_call_index = max(call_indices) if len(call_indices) > 0 else -1
            for idx, call_record in enumerate(call_records):
                reward = float(call_record["genver_loc_reward_quality"]) if call_indices[idx] == last_call_index else 0.0
                call_record["genver_loc_reward_raw"] = reward
                call_record["genver_loc_reward"] = reward
                sample_rewards[idx] = reward
        rewards.append(sample_rewards)
    return rewards


def compute_loc_verifier_rewards_batched(
    data_sources: Sequence[str],
    loc_verifier_call_records: Sequence[dict] | Iterable[dict],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
    parent_row_indices: Sequence[int] | Iterable[int] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Verifier objective: usefulness of feedback only.

    For each verifier call tied to call_index=t:
        delta_t = pq_true(t+1) - pq_true(t)
        raw_t = 1.0 if delta_t > eps
                0.0 if delta_t < -eps
                0.1 otherwise

    Invalid rows (missing current/next call eval) get 0.0.
    Per-sample normalization is over valid rows only.
    """
    del data_sources, ground_truths, extra_infos
    eps = 1e-8
    loc_eval_lookup_raw = kwargs.get("_genver_loc_eval_lookup", {})
    if isinstance(loc_eval_lookup_raw, dict):
        loc_eval_lookup = loc_eval_lookup_raw
    else:
        loc_eval_lookup = {}

    call_records: list[dict] = []
    for record in loc_verifier_call_records:
        if isinstance(record, dict):
            call_records.append(record)
        else:
            call_records.append({})

    if isinstance(parent_row_indices, np.ndarray):
        parent_row_index_list = parent_row_indices.tolist()
    elif isinstance(parent_row_indices, (list, tuple)):
        parent_row_index_list = list(parent_row_indices)
    else:
        parent_row_index_list = list(range(len(call_records)))
    if len(parent_row_index_list) < len(call_records):
        parent_row_index_list.extend(list(range(len(parent_row_index_list), len(call_records))))

    format_valid_flags = _build_format_valid_flags(solution_strs, len(call_records))
    eval_results: list[dict[str, Any] | None] = [None] * len(call_records)
    raw_rewards: list[float] = [0.0] * len(call_records)
    rewards: list[float] = [0.0] * len(call_records)
    for idx, call_record in enumerate(call_records):
        call_record["genver_loc_verifier_delta_pq"] = 0.0
        call_record["genver_loc_verifier_usefulness"] = 0.0
        call_record["genver_loc_verifier_reward_raw"] = 0.0
        call_record["genver_loc_verifier_reward"] = 0.0
        call_record["genver_loc_verifier_missing_next_call"] = False
        call_record["genver_loc_verifier_valid_for_reward"] = False
        call_record["genver_loc_verifier_feedback_valid_for_reward"] = bool(
            call_record.get("genver_loc_verifier_feedback_valid_for_reward", True)
        )
        call_record["genver_loc_verifier_feedback_has_effect"] = bool(
            call_record.get("genver_loc_verifier_feedback_has_effect", False)
        )
        call_record["genver_loc_verifier_feedback_has_duplicate_add_existing"] = bool(
            call_record.get("genver_loc_verifier_feedback_has_duplicate_add_existing", False)
        )
        call_record["genver_loc_verifier_feedback_has_disallowed_remove"] = bool(
            call_record.get("genver_loc_verifier_feedback_has_disallowed_remove", False)
        )
        call_record["genver_loc_verifier_feedback_has_invalid_remove"] = bool(
            call_record.get("genver_loc_verifier_feedback_has_invalid_remove", False)
        )
        call_record["genver_loc_verifier_feedback_has_remove_add_duplicate"] = bool(
            call_record.get("genver_loc_verifier_feedback_has_remove_add_duplicate", False)
        )
        call_record["genver_loc_verifier_feedback_duplicate_add_existing_count"] = int(
            _safe_int(call_record.get("genver_loc_verifier_feedback_duplicate_add_existing_count", 0), 0)
        )
        call_record["genver_loc_verifier_feedback_disallowed_remove_count"] = int(
            _safe_int(call_record.get("genver_loc_verifier_feedback_disallowed_remove_count", 0), 0)
        )
        call_record["genver_loc_verifier_feedback_invalid_remove_count"] = int(
            _safe_int(call_record.get("genver_loc_verifier_feedback_invalid_remove_count", 0), 0)
        )
        call_record["genver_loc_verifier_feedback_remove_add_duplicate_count"] = int(
            _safe_int(call_record.get("genver_loc_verifier_feedback_remove_add_duplicate_count", 0), 0)
        )
        if not format_valid_flags[idx]:
            continue

        parent_row_index = _safe_int(
            parent_row_index_list[idx] if idx < len(parent_row_index_list) else idx,
            idx,
        )
        call_index = _safe_int(call_record.get("call_index", idx), idx)
        loc_eval = loc_eval_lookup.get(f"{int(parent_row_index)}:{int(call_index)}")
        if not isinstance(loc_eval, dict):
            loc_eval_from_record = call_record.get("genver_loc_eval")
            if isinstance(loc_eval_from_record, dict):
                loc_eval = loc_eval_from_record
        pq_true = _clip01(loc_eval.get("pq", 0.0)) if isinstance(loc_eval, dict) else 0.0
        prediction_diagnostics: list[dict[str, Any]] = []
        if isinstance(loc_eval, dict):
            raw_prediction_diagnostics = loc_eval.get("prediction_diagnostics", [])
            if isinstance(raw_prediction_diagnostics, (list, tuple)):
                prediction_diagnostics = [
                    pred for pred in raw_prediction_diagnostics if isinstance(pred, dict)
                ]
        eval_result = {
            "pq": float(pq_true),
            "num_sam_boxes": int(loc_eval.get("num_sam_boxes", 0)) if isinstance(loc_eval, dict) else 0,
            "num_loc_boxes": int(loc_eval.get("num_loc_boxes", 0)) if isinstance(loc_eval, dict) else 0,
            "tp": int(loc_eval.get("tp", 0)) if isinstance(loc_eval, dict) else 0,
            "fp": int(loc_eval.get("fp", 0)) if isinstance(loc_eval, dict) else 0,
            "fn": int(loc_eval.get("fn", 0)) if isinstance(loc_eval, dict) else 0,
            "prediction_diagnostics": prediction_diagnostics,
        }
        eval_results[idx] = dict(eval_result)
        call_record["genver_sam3_eval"] = dict(eval_result)

    parent_to_indices: dict[int, list[int]] = {}
    for idx in range(len(call_records)):
        parent_row_index = _safe_int(
            parent_row_index_list[idx] if idx < len(parent_row_index_list) else idx,
            idx,
        )
        parent_to_indices.setdefault(int(parent_row_index), []).append(idx)

    for parent_row_index, parent_indices in parent_to_indices.items():
        if len(parent_indices) == 0:
            continue

        # Build call-index -> pq lookup from full loc eval lookup for this parent sample.
        # This includes the terminal call that may not have a corresponding verifier row.
        parent_call_pq: dict[int, float] = {}
        key_prefix = f"{int(parent_row_index)}:"
        for lookup_key, loc_eval in loc_eval_lookup.items():
            if not isinstance(lookup_key, str) or not lookup_key.startswith(key_prefix):
                continue
            call_suffix = lookup_key[len(key_prefix) :]
            try:
                call_index = int(call_suffix)
            except (TypeError, ValueError):
                continue
            if not isinstance(loc_eval, dict):
                continue
            parent_call_pq[call_index] = _clip01(loc_eval.get("pq", 0.0))

        # Keep verifier-row-local evals as a fallback for any missing lookup entries.
        for idx in parent_indices:
            call_index = _safe_int(call_records[idx].get("call_index", idx), idx)
            if call_index in parent_call_pq:
                continue
            eval_result = eval_results[idx]
            if eval_result is not None:
                parent_call_pq[call_index] = float(eval_result.get("pq", 0.0))

        valid_indices: list[int] = []
        for idx in parent_indices:
            if not format_valid_flags[idx]:
                continue
            if not bool(call_records[idx].get("genver_loc_verifier_feedback_valid_for_reward", True)):
                raw_rewards[idx] = 0.0
                continue
            call_index = _safe_int(call_records[idx].get("call_index", idx), idx)
            current_eval = eval_results[idx]
            if current_eval is None:
                continue
            current_pq = parent_call_pq.get(call_index)
            next_pq = parent_call_pq.get(call_index + 1)
            if current_pq is None or next_pq is None:
                call_records[idx]["genver_loc_verifier_missing_next_call"] = True
                raw_rewards[idx] = 0.0
                continue

            delta = float(next_pq - current_pq)
            if delta > eps:
                usefulness = 1.0
            elif delta < -eps:
                usefulness = 0.0
            else:
                usefulness = 0.1
            raw = float(_clip01(usefulness))
            call_records[idx]["genver_loc_verifier_delta_pq"] = float(delta)
            call_records[idx]["genver_loc_verifier_usefulness"] = float(usefulness)
            call_records[idx]["genver_loc_verifier_reward_raw"] = float(raw)
            call_records[idx]["genver_loc_verifier_valid_for_reward"] = True
            raw_rewards[idx] = raw
            valid_indices.append(idx)

        valid_count = len(valid_indices)
        valid_index_set = set(valid_indices)
        if valid_count > 0:
            reward_scale = 1.0 / float(valid_count)
            for idx in valid_indices:
                call_records[idx]["genver_loc_verifier_reward"] = float(raw_rewards[idx] * reward_scale)
        for idx in parent_indices:
            if idx not in valid_index_set:
                call_records[idx]["genver_loc_verifier_reward"] = 0.0

    for idx, call_record in enumerate(call_records):
        rewards[idx] = float(call_record.get("genver_loc_verifier_reward", 0.0))
    return rewards


def compute_answer_logic_verifier_rewards_batched(
    data_sources: Sequence[str],
    answer_logic_verifier_call_records: Sequence[dict] | Iterable[dict],
    answer_call_records: Sequence[Sequence[dict]] | Iterable[Sequence[dict]],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Iterable[dict],
    parent_row_indices: Sequence[int] | Iterable[int] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Answer logic self-verifier reward based on answer improvement.

    For a verifier row tied to answer_call_index=t:
      - Numeric targets/predictions: compare absolute error to GT.
          err_current = |pred_t - gt|
          err_next = |pred_{t+1} - gt|
          err_next < err_current - eps -> 1.0
          |err_next - err_current| <= eps -> 0.3
          err_next > err_current + eps -> 0.0
      - Non-numeric fallback:
          delta = answer_reward(t+1) - answer_reward(t)
          delta > eps      -> 1.0
          abs(delta) <= eps -> 0.3
          delta < -eps     -> 0.0

    High-score unchanged bonus:
      if abs(answer_reward(t+1) - answer_reward(t)) <= eps and answer_reward(t) > 0.7 -> 1.0

    Rewards are normalized per sample over valid verifier rows only.
    """
    del data_sources, kwargs
    eps = 1e-8

    call_records: list[dict] = []
    for record in answer_logic_verifier_call_records:
        if isinstance(record, dict):
            call_records.append(record)
        else:
            call_records.append({})

    if isinstance(answer_call_records, np.ndarray):
        answer_call_records_list = answer_call_records.tolist()
    else:
        answer_call_records_list = list(answer_call_records)

    if isinstance(ground_truths, np.ndarray):
        ground_truth_list = [str(x) for x in ground_truths.tolist()]
    else:
        ground_truth_list = [str(x) for x in ground_truths]

    if isinstance(extra_infos, np.ndarray):
        extra_info_list = extra_infos.tolist()
    else:
        extra_info_list = list(extra_infos)

    if isinstance(parent_row_indices, np.ndarray):
        parent_row_index_list = parent_row_indices.tolist()
    elif isinstance(parent_row_indices, (list, tuple)):
        parent_row_index_list = list(parent_row_indices)
    else:
        parent_row_index_list = list(range(len(call_records)))
    if len(parent_row_index_list) < len(call_records):
        parent_row_index_list.extend(list(range(len(parent_row_index_list), len(call_records))))

    rewards: list[float] = [0.0] * len(call_records)
    raw_rewards: list[float] = [0.0] * len(call_records)

    for idx, call_record in enumerate(call_records):
        call_record["genver_answer_logic_verifier_delta_answer_reward"] = 0.0
        call_record["genver_answer_logic_verifier_answer_reward_current"] = 0.0
        call_record["genver_answer_logic_verifier_answer_reward_next"] = 0.0
        call_record["genver_answer_logic_verifier_is_numeric_pair"] = False
        call_record["genver_answer_logic_verifier_err_current"] = 0.0
        call_record["genver_answer_logic_verifier_err_next"] = 0.0
        call_record["genver_answer_logic_verifier_delta_err"] = 0.0
        call_record["genver_answer_logic_verifier_reward_raw"] = 0.0
        call_record["genver_answer_logic_verifier_reward"] = 0.0
        call_record["genver_answer_logic_verifier_valid_for_reward"] = False
        call_record["genver_answer_logic_verifier_missing_next_call"] = False
        call_record["genver_answer_logic_verifier_high_score_unchanged_bonus"] = False

    parent_to_indices: dict[int, list[int]] = {}
    for idx in range(len(call_records)):
        parent_row_index = _safe_int(
            parent_row_index_list[idx] if idx < len(parent_row_index_list) else idx,
            idx,
        )
        parent_to_indices.setdefault(int(parent_row_index), []).append(idx)

    for parent_row_index, parent_indices in parent_to_indices.items():
        if len(parent_indices) == 0:
            continue
        gt = (
            ground_truth_list[parent_row_index]
            if 0 <= parent_row_index < len(ground_truth_list)
            else ""
        )
        extra_info = (
            extra_info_list[parent_row_index]
            if 0 <= parent_row_index < len(extra_info_list)
            and isinstance(extra_info_list[parent_row_index], dict)
            else {}
        )
        answer_type = extra_info.get("answer_type") if isinstance(extra_info, dict) else None

        # Build answer-call-index -> answer reward lookup for this sample.
        # Any missing/invalid call entries simply do not contribute to the lookup.
        answer_calls_for_sample: list[dict] = []
        any_idx = parent_indices[0]
        if 0 <= any_idx < len(answer_call_records_list):
            raw_answer_calls = answer_call_records_list[any_idx]
            if isinstance(raw_answer_calls, np.ndarray):
                raw_answer_calls = raw_answer_calls.tolist()
            if isinstance(raw_answer_calls, (list, tuple)):
                answer_calls_for_sample = [x for x in raw_answer_calls if isinstance(x, dict)]

        call_reward_lookup: dict[int, float] = {}
        call_pred_numeric_lookup: dict[int, float | None] = {}
        for answer_call in answer_calls_for_sample:
            call_index = _safe_int(answer_call.get("call_index", -1), -1)
            if call_index < 0:
                continue
            solution_str = str(answer_call.get("answer_solution_str", "") or "")
            if not solution_str:
                latest_bbox_block = str(answer_call.get("answer_latest_bbox_block", "") or "")
                answer_output_text = str(answer_call.get("answer_output_text", "") or "")
                solution_str = f"{latest_bbox_block}\n{answer_output_text}".strip()
            call_reward_lookup[call_index] = float(compute_score(solution_str, gt, answer_type))
            predicted_answer = _extract_answer(solution_str)
            call_pred_numeric_lookup[call_index] = _extract_float(predicted_answer or "")

        gt_numeric = _extract_float(gt)

        valid_indices: list[int] = []
        for idx in parent_indices:
            call_record = call_records[idx]
            parse_valid = bool(call_record.get("logic_feedback_parse_valid", False))
            feedback_valid_for_reward = bool(call_record.get("logic_feedback_valid_for_reward", parse_valid))
            answer_call_index = _safe_int(call_record.get("answer_call_index", -1), -1)
            if not parse_valid or not feedback_valid_for_reward or answer_call_index < 0:
                raw_rewards[idx] = 0.0
                continue

            current_score = call_reward_lookup.get(answer_call_index, None)
            next_score = call_reward_lookup.get(answer_call_index + 1, None)
            if current_score is None or next_score is None:
                call_record["genver_answer_logic_verifier_missing_next_call"] = True
                raw_rewards[idx] = 0.0
                continue

            delta = float(next_score - current_score)
            current_pred_numeric = call_pred_numeric_lookup.get(answer_call_index, None)
            next_pred_numeric = call_pred_numeric_lookup.get(answer_call_index + 1, None)
            is_numeric_pair = (
                gt_numeric is not None and current_pred_numeric is not None and next_pred_numeric is not None
            )
            err_current = 0.0
            err_next = 0.0
            delta_err = 0.0

            if is_numeric_pair:
                err_current = float(abs(current_pred_numeric - gt_numeric))
                err_next = float(abs(next_pred_numeric - gt_numeric))
                delta_err = float(err_current - err_next)
                if err_next < err_current - eps:
                    raw = 1.0
                elif abs(err_next - err_current) <= eps:
                    raw = 0.3
                else:
                    raw = 0.0
            else:
                if delta > eps:
                    raw = 1.0
                elif delta < -eps:
                    raw = 0.0
                else:
                    raw = 0.3

            high_score_unchanged_bonus = abs(delta) <= eps and current_score > 0.7
            if high_score_unchanged_bonus:
                raw = 1.0

            call_record["genver_answer_logic_verifier_delta_answer_reward"] = float(delta)
            call_record["genver_answer_logic_verifier_answer_reward_current"] = float(current_score)
            call_record["genver_answer_logic_verifier_answer_reward_next"] = float(next_score)
            call_record["genver_answer_logic_verifier_is_numeric_pair"] = bool(is_numeric_pair)
            call_record["genver_answer_logic_verifier_err_current"] = float(err_current)
            call_record["genver_answer_logic_verifier_err_next"] = float(err_next)
            call_record["genver_answer_logic_verifier_delta_err"] = float(delta_err)
            call_record["genver_answer_logic_verifier_high_score_unchanged_bonus"] = bool(
                high_score_unchanged_bonus
            )
            call_record["genver_answer_logic_verifier_reward_raw"] = float(_clip01(raw))
            call_record["genver_answer_logic_verifier_valid_for_reward"] = True
            raw_rewards[idx] = float(_clip01(raw))
            valid_indices.append(idx)

        valid_count = len(valid_indices)
        valid_set = set(valid_indices)
        if valid_count > 0:
            scale = 1.0 / float(valid_count)
            for idx in valid_indices:
                call_records[idx]["genver_answer_logic_verifier_reward"] = float(raw_rewards[idx] * scale)
        for idx in parent_indices:
            if idx not in valid_set:
                call_records[idx]["genver_answer_logic_verifier_reward"] = 0.0

    for idx, call_record in enumerate(call_records):
        rewards[idx] = float(call_record.get("genver_answer_logic_verifier_reward", 0.0))
    return rewards


__all__ = [
    "compute_score",
    "compute_score_batched",
    "compute_score_answer_only",
    "compute_score_answer_only_batched",
    "compute_loc_call_rewards_batched",
    "compute_loc_verifier_rewards_batched",
    "compute_answer_logic_verifier_rewards_batched",
]

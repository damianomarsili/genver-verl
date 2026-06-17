from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from google import genai
from google.genai import types as genai_types
from PIL import Image

from training.outcome_reward import (
    _box_1000_to_xyxy_pixels,
    _build_format_valid_flags,
    _extract_bbox_2d_entries_from_call_record,
    _extract_original_images_from_raw_prompt,
    _extract_query_text_from_raw_prompt,
    _is_strict_bbox_reason_answer_format,
    _is_strict_reason_answer_format,
    _normalize_cuda_device,
    _normalize_label,
    _pq_match_boxes,
    _resolve_sam3_worker_devices,
    _run_sam3_tasks_sharded,
)


SAFETY_CATEGORY_NAMES: Sequence[str] = (
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_MAX_WORKERS = 1
DEFAULT_TIMEOUT_S = 180
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_SLEEP_S = 5.0
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_STRONG_ANSWER_THRESHOLD = 0.7
DEFAULT_GLOBAL_MAX_INFLIGHT = 1
DEFAULT_GLOBAL_SLOT_WAIT_S = 900.0
DEFAULT_GLOBAL_SLOT_DIR = Path(__file__).resolve().parents[1] / ".cache" / "genver_gemini_slots"
ANSWER_BLOCK_PATTERN = re.compile(r"(?is)<answer>\s*(?P<body>.*?)\s*</answer>")


_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}
_RESPONSE_CACHE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _gemini_log(message: str) -> None:
    with _PRINT_LOCK:
        print(f"[gemini_reward] {message}", flush=True)


def _gemini_debug_io(config: dict[str, Any], header: str, payload: str) -> None:
    if not _coerce_bool(config.get("debug_print_io", False), False):
        return
    with _PRINT_LOCK:
        print(f"[gemini_reward][debug] {header}", flush=True)
        print(payload, flush=True)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    return _safe_float(value, default)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


def _load_api_key_from_file(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    candidate = Path(str(path_value).strip()).expanduser()
    try:
        resolved = candidate if candidate.is_absolute() else candidate.resolve()
    except FileNotFoundError:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    key = resolved.read_text(encoding="utf-8").strip()
    return key or None


def _resolve_api_key(api_key: Optional[str], api_key_file: Optional[str]) -> Optional[str]:
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    from_file = _load_api_key_from_file(api_key_file)
    if from_file:
        return from_file
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def _default_safety_settings() -> Optional[list[Any]]:
    settings = []
    for category in SAFETY_CATEGORY_NAMES:
        try:
            settings.append(genai_types.SafetySetting(category=category, threshold="BLOCK_NONE"))
        except Exception:
            continue
    return settings or None


def _build_http_options(timeout_s: int, http_api_version: Optional[str]) -> Any:
    # google-genai expects HttpOptions.timeout in milliseconds.
    timeout_ms = max(1, int(timeout_s)) * 1000
    return genai_types.HttpOptions(
        api_version=http_api_version or None,
        timeout=timeout_ms,
    )


def _create_genai_client(
    *,
    api_key: Optional[str],
    api_key_file: Optional[str],
    timeout_s: int,
    http_api_version: Optional[str],
) -> genai.Client:
    client_kwargs: dict[str, Any] = {
        "http_options": _build_http_options(timeout_s, http_api_version),
    }
    resolved_key = _resolve_api_key(api_key, api_key_file)
    if not resolved_key:
        raise ValueError("Provide GEMINI_API_KEY (or GOOGLE_API_KEY).")
    client_kwargs["api_key"] = resolved_key

    return genai.Client(**client_kwargs)


def _get_thread_client(config: dict[str, Any]) -> tuple[genai.Client, Optional[list[Any]]]:
    client = _create_genai_client(
        api_key=config["api_key"],
        api_key_file=config["api_key_file"],
        timeout_s=int(config["timeout_s"]),
        http_api_version=config["http_api_version"],
    )
    safety_settings = _default_safety_settings()
    return client, safety_settings


def _reset_thread_client() -> None:
    return None


def _load_prompt_template(
    prompt_path: Optional[str],
    *,
    default_filename: str = "gemini_answer_grader_instructions.txt",
) -> str:
    if prompt_path:
        path = Path(prompt_path)
    else:
        path = Path(__file__).resolve().parent / "prompts" / default_filename
    prompt_text = path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Gemini answer grader prompt is empty: {path}")
    return prompt_text


def _part_from_text(text: str) -> Any:
    return genai_types.Part(text=text)


def _part_from_image(image: Image.Image) -> Any:
    if hasattr(genai_types.Part, "from_image"):
        return genai_types.Part.from_image(image=image)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()

    if hasattr(genai_types.Part, "from_bytes"):
        return genai_types.Part.from_bytes(data=data, mime_type="image/png")  # type: ignore[attr-defined]

    if hasattr(genai_types.Part, "InlineData"):
        return genai_types.Part(  # type: ignore[call-arg]
            inline_data=genai_types.Part.InlineData(data=data, mime_type="image/png")
        )

    raise AttributeError("Installed google-genai library cannot construct an image part.")


def _ensure_slot_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def _acquire_gemini_global_slot(config: dict[str, Any], *, tag: str) -> Any:
    max_inflight = max(
        1,
        _safe_int(
            config.get("global_max_inflight", DEFAULT_GLOBAL_MAX_INFLIGHT),
            DEFAULT_GLOBAL_MAX_INFLIGHT,
        ),
    )
    slot_wait_s = max(
        1.0,
        _safe_float(
            config.get("global_slot_wait_s", DEFAULT_GLOBAL_SLOT_WAIT_S),
            DEFAULT_GLOBAL_SLOT_WAIT_S,
        ),
    )
    slot_dir = Path(
        str(
            config.get("global_slot_dir")
            or os.getenv("GEMINI_GLOBAL_SLOT_DIR")
            or DEFAULT_GLOBAL_SLOT_DIR
        )
    )
    _ensure_slot_dir(slot_dir)
    handles: list[Any] = []
    deadline = time.monotonic() + slot_wait_s
    try:
        while True:
            for slot_idx in range(max_inflight):
                slot_path = slot_dir / f"slot_{slot_idx}.lock"
                handle = open(slot_path, "a+b")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    handle.seek(0)
                    handle.truncate()
                    handle.write(
                        f"{os.getpid()}:{threading.get_ident()}:{tag}\n".encode(
                            "utf-8"
                        )
                    )
                    handle.flush()
                    handles.append(handle)
                    yield
                    return
                except BlockingIOError:
                    handle.close()
                    continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Gemini global slot after {slot_wait_s:.1f}s"
                )
            time.sleep(0.2 + random.random() * 0.1)
    finally:
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass


def _resize_longest_side(image: Image.Image, longest_side: int) -> Image.Image:
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


def _extract_json_dict(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    candidates: list[str] = [text]
    if text.startswith("```"):
        stripped = text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(text[first_brace : last_brace + 1])

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _response_text_from_obj(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, (list, tuple)):
        return str(text or "")
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                chunks.append(part_text)
    if chunks:
        return "\n".join(chunks)
    return str(text or "")


def _response_preview(text: str, limit: int = 200) -> str:
    text = str(text or "")
    return text[:limit]


def _extract_json_payload_from_response(response: Any) -> tuple[dict[str, Any], str]:
    # Keep parsing simple: trust structured payload first, then exact JSON text.
    parsed_obj = getattr(response, "parsed", None)
    if isinstance(parsed_obj, dict):
        return parsed_obj, _response_text_from_obj(response)
    if hasattr(parsed_obj, "model_dump"):
        try:
            dumped = parsed_obj.model_dump()
            if isinstance(dumped, dict):
                return dumped, _response_text_from_obj(response)
        except Exception:
            pass
    if isinstance(parsed_obj, str):
        parsed_from_str = _extract_json_dict(parsed_obj)
        if parsed_from_str:
            return parsed_from_str, _response_text_from_obj(response)

    response_text = _response_text_from_obj(response)
    return _extract_json_dict(response_text), response_text


def _extract_numeric_json_field(text: str, field: str) -> Optional[float]:
    text = str(text or "")
    if not text or not field:
        return None
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _extract_string_json_field(text: str, field: str) -> Optional[str]:
    text = str(text or "")
    if not text or not field:
        return None
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"')
    match = pattern.search(text)
    if match is None:
        return None
    raw_value = match.group(1)
    try:
        return json.loads(f'"{raw_value}"')
    except Exception:
        return raw_value


def _extract_partial_logic_teacher_payload(response_text: str) -> Optional[dict[str, Any]]:
    text = str(response_text or "")
    if not text:
        return None
    current_answer_score = _extract_numeric_json_field(text, "current_answer_score")
    self_edit_score = _extract_numeric_json_field(text, "self_edit_score")
    if current_answer_score is None and self_edit_score is None:
        return None

    payload: dict[str, Any] = {
        "current_answer_score": float(current_answer_score if current_answer_score is not None else 0.0),
        "self_edit_score": float(self_edit_score if self_edit_score is not None else 0.0),
        "self_edit_reason": str(_extract_string_json_field(text, "self_edit_reason") or "").strip(),
        "teacher_edits": [],
    }

    edits: list[str] = []
    edits_block_match = re.search(r'"teacher_edits"\s*:\s*\[(?P<body>.*?)\]', text, flags=re.DOTALL)
    if edits_block_match is not None:
        body = edits_block_match.group("body")
        for raw_match in re.finditer(r'"((?:\\.|[^"\\])*)"', body):
            raw_item = raw_match.group(1)
            try:
                decoded = json.loads(f'"{raw_item}"')
            except Exception:
                decoded = raw_item
            decoded = str(decoded or "").strip()
            if decoded:
                edits.append(decoded)
    if not edits:
        edits = [
            str(match.group(0)).strip()
            for match in re.finditer(r"(?im)^EDIT_STEP\s+\d+\s*:\s*.+$", text)
        ]
    payload["teacher_edits"] = edits[:2]
    return payload


def _build_thinking_config_for_structured_output() -> Optional[Any]:
    # Gemini 3 guidance: use thinking_level (not legacy thinking_budget).
    # We keep thoughts hidden and use minimal thinking to reduce output churn.
    try:
        return genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_level="minimal",
        )
    except Exception:
        return None


def _has_required_fields(payload: dict[str, Any], required_fields: Sequence[str]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in required_fields:
        if key not in payload or payload.get(key, None) is None:
            return False
    return True


def _normalize_score_to_01(value: Any) -> float:
    """Normalize Gemini scores to [0,1], preferring 0-10 scale."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    # Compatibility fallback when model emits [0,1] directly.
    if 0.0 <= numeric <= 1.0:
        return round(numeric, 3)
    return round(max(0.0, min(10.0, numeric)) / 10.0, 3)


def _normalize_response_dict(payload: dict[str, Any]) -> dict[str, Any]:
    score = _normalize_score_to_01(payload.get("score", 0.0))
    has_explicit_score = "score" in payload and payload.get("score", None) is not None
    verdict = str(payload.get("verdict", "") or "").strip().lower()
    if verdict not in {"correct", "mostly_correct", "partially_correct", "incorrect"}:
        if has_explicit_score:
            if score >= 0.95:
                verdict = "correct"
            elif score >= 0.75:
                verdict = "mostly_correct"
            elif score >= 0.4:
                verdict = "partially_correct"
            else:
                verdict = "incorrect"
        else:
            verdict = ""
    reason = str(payload.get("reason", "") or "").strip()
    return {
        "score": score,
        "verdict": verdict,
        "reason": reason,
    }


def _request_cache_key(query: str, candidate_response: str, image_fingerprints: Sequence[str], model: str) -> str:
    payload = json.dumps(
        {
            "query": query,
            "candidate_response": candidate_response,
            "images": list(image_fingerprints),
            "model": model,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_request_prompt(template: str, query: str, candidate_response: str) -> str:
    core = template.format(
        query=query.strip(),
        candidate_response=str(candidate_response or "").strip(),
    )
    return (
        f"{core.rstrip()}\n\n"
        "FINAL OUTPUT RULE: Output ONLY the JSON object and NO OTHER TEXT."
    )


def _extract_final_answer_text(candidate_response: str) -> str:
    text = str(candidate_response or "")
    match = ANSWER_BLOCK_PATTERN.search(text)
    if match is None:
        return "<missing>"
    return str(match.group("body") or "").strip() or "<empty>"


def _build_reason_answer_format_valid_flags(solution_strs: Sequence[str], num_samples: int) -> list[bool]:
    if isinstance(solution_strs, np.ndarray):
        values = solution_strs.tolist()
    elif isinstance(solution_strs, (list, tuple)):
        values = list(solution_strs)
    else:
        values = []
    flags = [False] * num_samples
    for idx in range(min(num_samples, len(values))):
        flags[idx] = _is_strict_reason_answer_format(str(values[idx] or ""))
    return flags


def _is_any_supported_genver_format(text: str) -> bool:
    candidate = str(text or "")
    return bool(
        _is_strict_reason_answer_format(candidate)
        or _is_strict_bbox_reason_answer_format(candidate)
    )


def _build_any_supported_format_valid_flags(solution_strs: Sequence[str], num_samples: int) -> list[bool]:
    if isinstance(solution_strs, np.ndarray):
        values = solution_strs.tolist()
    elif isinstance(solution_strs, (list, tuple)):
        values = list(solution_strs)
    else:
        values = []
    flags = [False] * num_samples
    for idx in range(min(num_samples, len(values))):
        flags[idx] = _is_any_supported_genver_format(str(values[idx] or ""))
    return flags


def _extract_final_answer_solution_from_call_records(answer_calls: Any) -> str:
    if isinstance(answer_calls, np.ndarray):
        answer_calls = answer_calls.tolist()
    if isinstance(answer_calls, dict):
        calls = [answer_calls]
    elif isinstance(answer_calls, (list, tuple)):
        calls = list(answer_calls)
    else:
        calls = []

    best_call: Optional[dict[str, Any]] = None
    best_call_index = -1
    for pos, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        call_index = _safe_int(call.get("call_index", pos), pos)
        if best_call is None or call_index >= best_call_index:
            best_call = call
            best_call_index = call_index
    if best_call is None:
        return ""
    solution = str(best_call.get("answer_solution_str", "") or "")
    if solution:
        return solution
    return str(best_call.get("answer_output_text", "") or "")


def _build_logic_teacher_prompt(
    template: str,
    *,
    query: str,
    detected_objects: str,
    current_answer: str,
    proposed_self_edits: str,
) -> str:
    core = template.format(
        query=query.strip(),
        detected_objects=str(detected_objects or "").strip(),
        current_answer=str(current_answer or "").strip(),
        proposed_self_edits=str(proposed_self_edits or "").strip(),
    )
    return (
        f"{core.rstrip()}\n\n"
        "FINAL OUTPUT RULE: Output ONLY the JSON object and NO OTHER TEXT."
    )


def _generate_gemini_judgment(
    *,
    config: dict[str, Any],
    prompt_template: str,
    query: str,
    candidate_response: str,
    images: Sequence[Image.Image],
) -> dict[str, Any]:
    prompt = _build_request_prompt(prompt_template, query, candidate_response)
    resized_images = [_resize_longest_side(image.convert("RGB"), int(config["max_image_side"])) for image in images]
    contents = [_part_from_text(prompt)] + [_part_from_image(image) for image in resized_images]
    _gemini_debug_io(
        config,
        header="answer-grade request",
        payload=(
            f"model={config['model']}\n"
            f"query:\n{query}\n\n"
            f"extracted_final_answer={_extract_final_answer_text(candidate_response)!r}\n\n"
            f"candidate_response:\n{candidate_response}\n\n"
            f"image_count={len(images)}\n"
            f"prompt_to_gemini:\n{prompt}\n"
        ),
    )
    client, safety_settings = _get_thread_client(config)
    answer_schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "reason": {"type": "string"},
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
            },
        },
        "required": ["score"],
    }
    def _build_answer_generation_config(
        schema_mode: str,
        safety: Any,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_output_tokens": int(config["max_output_tokens"]),
            "response_mime_type": "application/json",
        }
        thinking_config = _build_thinking_config_for_structured_output()
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        if schema_mode == "response_schema":
            config_kwargs["response_schema"] = answer_schema
        else:
            config_kwargs["response_json_schema"] = answer_schema
        cfg = genai_types.GenerateContentConfig(**config_kwargs)
        if safety is not None:
            cfg.safety_settings = safety
        return cfg

    config_enabled = True
    schema_mode = "response_json_schema"
    last_error = ""
    for attempt in range(int(config["max_retries"])):
        try:
            request_kwargs: dict[str, Any] = {
                "model": str(config["model"]),
                "contents": contents,
            }
            if config_enabled:
                request_kwargs["config"] = _build_answer_generation_config(
                    schema_mode=schema_mode,
                    safety=safety_settings,
                )
            with _acquire_gemini_global_slot(config, tag="answer_reward"):
                response = client.models.generate_content(**request_kwargs)
            parsed, response_text = _extract_json_payload_from_response(response)
            if not _has_required_fields(parsed, ("score",)):
                score_from_partial = _extract_numeric_json_field(response_text, "score")
                if score_from_partial is not None:
                    parsed = dict(parsed or {})
                    parsed["score"] = score_from_partial
                    _gemini_log(
                        f"model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                        "used partial numeric salvage for score."
                    )
            if not parsed:
                last_error = f"Malformed Gemini JSON: {response_text[:500]}"
                _gemini_log(
                    f"malformed response from model={config['model']} "
                    f"attempt={attempt + 1}/{int(config['max_retries'])}; retrying. "
                    f"response_preview={_response_preview(response_text)!r}"
                )
                if attempt < int(config["max_retries"]) - 1:
                    time.sleep(float(config["retry_sleep_s"]))
                    continue
            if not _has_required_fields(parsed, ("score",)):
                last_error = f"Invalid Gemini schema payload: {response_text[:500]}"
                _gemini_log(
                    f"invalid schema payload from model={config['model']} "
                    f"attempt={attempt + 1}/{int(config['max_retries'])}; retrying. "
                    f"response_preview={_response_preview(response_text)!r}"
                )
                if attempt < int(config["max_retries"]) - 1:
                    time.sleep(float(config["retry_sleep_s"]))
                    continue
            normalized = _normalize_response_dict(
                {
                    "score": parsed.get("score", 0.0),
                    "verdict": parsed.get("verdict", ""),
                    "reason": parsed.get("reason", ""),
                }
            )
            _gemini_debug_io(
                config,
                header="answer-grade response",
                payload=(
                    f"raw_response:\n{response_text}\n\n"
                    f"parsed_normalized:\n{json.dumps(normalized, ensure_ascii=False)}\n"
                ),
            )
            normalized["raw_text"] = response_text
            normalized["failed"] = False
            normalized["error"] = ""
            return normalized
        except TypeError as exc:
            message = str(exc)
            last_error = message
            _reset_thread_client()
            if "config" in message and config_enabled:
                _gemini_log(
                    f"model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                    f"hit unsupported config parameter; retrying without config. error={message}"
                )
                config_enabled = False
                continue
            _gemini_log(
                f"model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                f"failed with TypeError; retrying. error={message}"
            )
        except Exception as exc:
            last_error = str(exc)
            _reset_thread_client()
            if (
                config_enabled
                and schema_mode == "response_json_schema"
                and "response_json_schema" in last_error
            ):
                _gemini_log(
                    f"model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                    f"response_json_schema failed; retrying with response_schema fallback. error={last_error}"
                )
                schema_mode = "response_schema"
                continue
            _gemini_log(
                f"model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                f"failed; retrying. error={last_error}"
            )
        if attempt < int(config["max_retries"]) - 1:
            time.sleep(float(config["retry_sleep_s"]) * (1.5**attempt))

    _gemini_log(
        f"model={config['model']} exhausted retries; assigning reward 0. "
        f"last_error={last_error}"
    )
    return {
        "score": 0.0,
        "verdict": "incorrect",
        "reason": "",
        "raw_text": "",
        "failed": True,
        "error": last_error,
    }


def _normalize_logic_teacher_response_dict(payload: dict[str, Any]) -> dict[str, Any]:
    current_answer_score = _normalize_score_to_01(payload.get("current_answer_score", 0.0))
    self_edit_score = _normalize_score_to_01(payload.get("self_edit_score", 0.0))
    self_edit_reason = str(payload.get("self_edit_reason", "") or "").strip()
    teacher_edits_raw = payload.get("teacher_edits", [])
    if isinstance(teacher_edits_raw, str):
        teacher_edits = [
            line.strip()
            for line in teacher_edits_raw.splitlines()
            if line.strip()
        ]
    elif isinstance(teacher_edits_raw, (list, tuple)):
        teacher_edits = [str(line or "").strip() for line in teacher_edits_raw if str(line or "").strip()]
    else:
        teacher_edits = []
    teacher_edits = teacher_edits[:2]
    return {
        "current_answer_score": current_answer_score,
        "self_edit_score": self_edit_score,
        "self_edit_reason": self_edit_reason,
        "teacher_edits": teacher_edits,
    }


def _generate_gemini_logic_teacher_judgment(
    *,
    config: dict[str, Any],
    prompt_template: str,
    query: str,
    detected_objects: str,
    current_answer: str,
    proposed_self_edits: str,
    images: Sequence[Image.Image],
) -> dict[str, Any]:
    prompt = _build_logic_teacher_prompt(
        prompt_template,
        query=query,
        detected_objects=detected_objects,
        current_answer=current_answer,
        proposed_self_edits=proposed_self_edits,
    )
    resized_images = [_resize_longest_side(image.convert("RGB"), int(config["max_image_side"])) for image in images]
    contents = [_part_from_text(prompt)] + [_part_from_image(image) for image in resized_images]
    client, safety_settings = _get_thread_client(config)
    logic_schema = {
        "type": "object",
        "properties": {
            "self_edit_reason": {
                "type": "string",
                "maxLength": 240,
            },
            "teacher_edits": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "string",
                    "maxLength": 220,
                },
            },
            "current_answer_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 10.0,
                "multipleOf": 0.01,
            },
            "self_edit_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 10.0,
                "multipleOf": 0.01,
            },
        },
        "required": [
            "current_answer_score",
            "self_edit_score",
            "self_edit_reason",
            "teacher_edits",
        ],
    }
    def _build_logic_generation_config(
        schema_mode: str,
        safety: Any,
    ) -> Any:
        config_kwargs: dict[str, Any] = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_output_tokens": max(512, int(config["max_output_tokens"])),
            "response_mime_type": "application/json",
        }
        thinking_config = _build_thinking_config_for_structured_output()
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        if schema_mode == "response_schema":
            config_kwargs["response_schema"] = logic_schema
        else:
            config_kwargs["response_json_schema"] = logic_schema
        cfg = genai_types.GenerateContentConfig(**config_kwargs)
        if safety is not None:
            cfg.safety_settings = safety
        return cfg

    config_enabled = True
    schema_mode = "response_json_schema"
    last_error = ""
    for attempt in range(int(config["max_retries"])):
        try:
            request_kwargs: dict[str, Any] = {
                "model": str(config["model"]),
                "contents": contents,
            }
            if config_enabled:
                request_kwargs["config"] = _build_logic_generation_config(
                    schema_mode=schema_mode,
                    safety=safety_settings,
                )
            with _acquire_gemini_global_slot(config, tag="logic_teacher"):
                response = client.models.generate_content(**request_kwargs)
            parsed, response_text = _extract_json_payload_from_response(response)
            if not parsed:
                partial_payload = _extract_partial_logic_teacher_payload(response_text)
                if partial_payload is not None:
                    normalized = _normalize_logic_teacher_response_dict(partial_payload)
                    normalized["raw_text"] = response_text
                    normalized["failed"] = False
                    normalized["error"] = ""
                    _gemini_log(
                        f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                        "recovered partial JSON payload; continuing without retry."
                    )
                    return normalized
                last_error = f"Malformed Gemini logic-teacher JSON: {response_text[:500]}"
                _gemini_log(
                    f"logic-teacher malformed response from model={config['model']} "
                    f"attempt={attempt + 1}/{int(config['max_retries'])}; retrying. "
                    f"response_preview={_response_preview(response_text)!r}"
                )
                if attempt < int(config["max_retries"]) - 1:
                    time.sleep(float(config["retry_sleep_s"]))
                    continue
            if not _has_required_fields(
                parsed,
                ("current_answer_score", "self_edit_score", "self_edit_reason", "teacher_edits"),
            ):
                partial_payload = _extract_partial_logic_teacher_payload(response_text)
                if partial_payload is not None:
                    normalized = _normalize_logic_teacher_response_dict(partial_payload)
                    normalized["raw_text"] = response_text
                    normalized["failed"] = False
                    normalized["error"] = ""
                    _gemini_log(
                        f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                        "recovered partial schema payload; continuing without retry."
                    )
                    return normalized
                last_error = f"Invalid Gemini logic-teacher schema payload: {response_text[:500]}"
                _gemini_log(
                    f"logic-teacher invalid schema payload from model={config['model']} "
                    f"attempt={attempt + 1}/{int(config['max_retries'])}; retrying. "
                    f"response_preview={_response_preview(response_text)!r}"
                )
                if attempt < int(config["max_retries"]) - 1:
                    time.sleep(float(config["retry_sleep_s"]))
                    continue
            normalized = _normalize_logic_teacher_response_dict(parsed)
            normalized["raw_text"] = response_text
            normalized["failed"] = False
            normalized["error"] = ""
            return normalized
        except TypeError as exc:
            message = str(exc)
            last_error = message
            _reset_thread_client()
            if "config" in message and config_enabled:
                _gemini_log(
                    f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                    f"hit unsupported config parameter; retrying without config. error={message}"
                )
                config_enabled = False
                continue
            _gemini_log(
                f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                f"failed with TypeError; retrying. error={message}"
            )
        except Exception as exc:
            last_error = str(exc)
            _reset_thread_client()
            if (
                config_enabled
                and schema_mode == "response_json_schema"
                and "response_json_schema" in last_error
            ):
                _gemini_log(
                    f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                    f"response_json_schema failed; retrying with response_schema fallback. error={last_error}"
                )
                schema_mode = "response_schema"
                continue
            _gemini_log(
                f"logic-teacher model={config['model']} attempt={attempt + 1}/{int(config['max_retries'])} "
                f"failed; retrying. error={last_error}"
            )
        if attempt < int(config["max_retries"]) - 1:
            time.sleep(float(config["retry_sleep_s"]) * (1.5**attempt))

    _gemini_log(
        f"logic-teacher model={config['model']} exhausted retries; assigning zeros. "
        f"last_error={last_error}"
    )
    return {
        "current_answer_score": 0.0,
        "self_edit_score": 0.0,
        "self_edit_reason": "",
        "teacher_edits": [],
        "raw_text": "",
        "failed": True,
        "error": last_error,
    }


def _build_reward_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(kwargs.get("gemini_model", os.getenv("GEMINI_REWARD_MODEL", DEFAULT_GEMINI_MODEL))),
        "api_key": kwargs.get("gemini_api_key", None),
        "api_key_file": kwargs.get(
            "gemini_api_key_file",
            os.getenv("GEMINI_API_KEY_FILE", None),
        ),
        "http_api_version": kwargs.get("gemini_http_api_version", None),
        "timeout_s": _safe_int(kwargs.get("gemini_timeout_s", DEFAULT_TIMEOUT_S), DEFAULT_TIMEOUT_S),
        "max_retries": max(1, _safe_int(kwargs.get("gemini_max_retries", DEFAULT_MAX_RETRIES), DEFAULT_MAX_RETRIES)),
        "retry_sleep_s": max(0.0, _safe_float(kwargs.get("gemini_retry_sleep_s", DEFAULT_RETRY_SLEEP_S), DEFAULT_RETRY_SLEEP_S)),
        "max_workers": max(1, _safe_int(kwargs.get("gemini_max_workers", DEFAULT_MAX_WORKERS), DEFAULT_MAX_WORKERS)),
        "max_output_tokens": max(32, _safe_int(kwargs.get("gemini_max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS)),
        "max_image_side": max(256, _safe_int(kwargs.get("gemini_max_image_side", 768), 768)),
        "global_max_inflight": max(
            1,
            _safe_int(
                kwargs.get(
                    "gemini_global_max_inflight",
                    os.getenv("GEMINI_GLOBAL_MAX_INFLIGHT", DEFAULT_GLOBAL_MAX_INFLIGHT),
                ),
                DEFAULT_GLOBAL_MAX_INFLIGHT,
            ),
        ),
        "global_slot_wait_s": max(
            1.0,
            _safe_float(
                kwargs.get(
                    "gemini_global_slot_wait_s",
                    os.getenv("GEMINI_GLOBAL_SLOT_WAIT_S", DEFAULT_GLOBAL_SLOT_WAIT_S),
                ),
                DEFAULT_GLOBAL_SLOT_WAIT_S,
            ),
        ),
        "global_slot_dir": str(
            kwargs.get("gemini_global_slot_dir", os.getenv("GEMINI_GLOBAL_SLOT_DIR", str(DEFAULT_GLOBAL_SLOT_DIR)))
            or DEFAULT_GLOBAL_SLOT_DIR
        ),
        "failure_mode": str(kwargs.get("gemini_failure_mode", "zero") or "zero").strip().lower(),
        "debug_print_io": _coerce_bool(kwargs.get("gemini_debug_print_io", False), False),
    }


def build_gemini_runtime_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _build_reward_config(kwargs)


def load_gemini_prompt_template(
    prompt_path: Optional[str],
    *,
    default_filename: str,
) -> str:
    return _load_prompt_template(prompt_path, default_filename=default_filename)


def generate_gemini_answer_judgment(
    *,
    config: dict[str, Any],
    prompt_template: str,
    query: str,
    candidate_response: str,
    images: Sequence[Image.Image],
) -> dict[str, Any]:
    return _generate_gemini_judgment(
        config=config,
        prompt_template=prompt_template,
        query=query,
        candidate_response=candidate_response,
        images=images,
    )


def generate_gemini_logic_teacher_judgment(
    *,
    config: dict[str, Any],
    prompt_template: str,
    query: str,
    detected_objects: str,
    current_answer: str,
    proposed_self_edits: str,
    images: Sequence[Image.Image],
) -> dict[str, Any]:
    return _generate_gemini_logic_teacher_judgment(
        config=config,
        prompt_template=prompt_template,
        query=query,
        detected_objects=detected_objects,
        current_answer=current_answer,
        proposed_self_edits=proposed_self_edits,
        images=images,
    )


def _extract_relevant_objects(extra_info: Any) -> list[str]:
    candidate = extra_info
    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()
    if isinstance(candidate, dict) and "relevant_objects" not in candidate:
        nested = candidate.get("extra_info")
        if isinstance(nested, dict) and "relevant_objects" in nested:
            candidate = nested

    relevant_objects = candidate.get("relevant_objects") if isinstance(candidate, dict) else candidate
    if isinstance(relevant_objects, np.ndarray):
        relevant_objects = relevant_objects.tolist()
    if isinstance(relevant_objects, str):
        stripped = relevant_objects.strip()
        if not stripped:
            relevant_objects = []
        else:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in stripped.split(",")]
            relevant_objects = parsed

    labels: list[str] = []
    if isinstance(relevant_objects, (list, tuple, set)):
        iterable = relevant_objects
    elif relevant_objects is None:
        iterable = []
    else:
        iterable = [relevant_objects]

    for value in iterable:
        label = _normalize_label(value)
        if label and label not in labels:
            labels.append(label)
    return labels


def _serialize_entries_and_targets_for_dedupe(
    entries: Sequence[dict[str, Any]],
    relevant_objects: Sequence[str],
) -> str:
    serialized_entries: list[list[Any]] = []
    for entry in entries:
        box = entry.get("box_1000")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        serialized_entries.append(
            [
                int(entry.get("image_index", 1)),
                str(entry.get("label", "")),
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            ]
        )
    payload = {
        "entries": serialized_entries,
        "relevant_objects": list(relevant_objects),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)


def _evaluate_loc_entries_with_relevant_objects(
    entries: Sequence[dict[str, Any]],
    originals: Sequence[tuple[Image.Image, str]],
    relevant_objects: Sequence[str],
    *,
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
        image_pos = int(entry.get("image_index", 1)) - 1
        if image_pos < 0 or image_pos >= len(originals):
            continue
        image, _ = originals[image_pos]
        predicted_label = _normalize_label(entry.get("label", ""))
        loc_boxes.append(
            {
                "image_index": int(entry.get("image_index", 1)),
                "predicted_label": predicted_label,
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

    relevant_labels = [label for label in relevant_objects if label]
    loc_by_image: dict[int, list[dict[str, Any]]] = {}
    for loc_box in loc_boxes:
        loc_by_image.setdefault(int(loc_box["image_index"]), []).append(loc_box)

    sam_by_image: dict[int, list[dict[str, Any]]] = {image_index: [] for image_index in loc_by_image.keys()}
    image_to_task_keys: dict[int, list[tuple[str, str]]] = {image_index: [] for image_index in loc_by_image.keys()}
    task_payloads: dict[tuple[str, str], tuple[Image.Image, str, str]] = {}
    if len(relevant_labels) > 0:
        for image_index in loc_by_image.keys():
            image_pos = image_index - 1
            if image_pos < 0 or image_pos >= len(originals):
                continue
            image, image_fingerprint = originals[image_pos]
            for label in relevant_labels:
                task_key = (image_fingerprint, label)
                image_to_task_keys[image_index].append(task_key)
                if task_key not in task_payloads:
                    task_payloads[task_key] = (image, label, image_fingerprint)

    sam3_task_results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    sam3_shard_metrics: dict[str, float] = {}
    if len(task_payloads) > 0:
        tasks = [
            (task_key, image, label, image_fingerprint)
            for task_key, (image, label, image_fingerprint) in task_payloads.items()
        ]
        sam3_task_results, sam3_shard_metrics = _run_sam3_tasks_sharded(
            tasks=tasks,
            devices=worker_devices,
            confidence_threshold=confidence_threshold,
            checkpoint_path=checkpoint_path,
            load_from_hf=load_from_hf,
        )

    for image_index, task_keys in image_to_task_keys.items():
        group_sam_boxes: list[dict[str, Any]] = []
        for task_key in task_keys:
            _, label = task_key
            for box_xyxy in sam3_task_results.get(task_key, []):
                group_sam_boxes.append(
                    {
                        "image_index": int(image_index),
                        "label": str(label),
                        "box_xyxy": tuple(float(x) for x in box_xyxy),
                    }
                )
        sam_by_image[image_index] = group_sam_boxes

    tp = 0
    fp = 0
    fn = 0
    sum_iou = 0.0
    diagnostics: list[dict[str, Any]] = []
    sam_index = 0

    for image_index, group_loc_boxes in loc_by_image.items():
        group_sam_boxes = sam_by_image.get(image_index, [])
        matches, iou_matrix = _pq_match_boxes(group_loc_boxes, group_sam_boxes)
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
                if column.shape[0] > 0:
                    best_iou = float(column.max())
                    best_loc_idx = int(column.argmax())
                    covered_loc_indices.add(best_loc_idx)

            matched_pair = match_by_sam.get(group_sam_idx)
            matched_loc_idx = matched_pair[0] if matched_pair is not None else -1
            diagnostics.append(
                {
                    "sam_index": sam_index,
                    "image_index": int(sam_box["image_index"]),
                    "label": str(sam_box["label"]),
                    "sam3_box_xyxy": tuple(float(x) for x in sam_box["box_xyxy"]),
                    "best_iou": float(best_iou),
                    "matched_iou": float(matched_pair[1]) if matched_pair is not None else 0.0,
                    "matched": matched_pair is not None,
                    "predicted_box_1000": (
                        tuple(float(x) for x in group_loc_boxes[best_loc_idx]["box_1000"])
                        if best_loc_idx >= 0
                        else None
                    ),
                    "predicted_box_xyxy": (
                        tuple(float(x) for x in group_loc_boxes[best_loc_idx]["box_xyxy"])
                        if best_loc_idx >= 0
                        else None
                    ),
                    "predicted_label": (
                        str(group_loc_boxes[best_loc_idx].get("predicted_label", ""))
                        if best_loc_idx >= 0
                        else ""
                    ),
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
            sam3_box_xyxy = (
                tuple(float(x) for x in group_sam_boxes[matched_sam_idx]["box_xyxy"])
                if matched_sam_idx >= 0 and matched_sam_idx < len(group_sam_boxes)
                else None
            )
            diagnostics.append(
                {
                    "sam_index": -1,
                    "image_index": int(loc_box["image_index"]),
                    "label": str(loc_box.get("predicted_label", "")),
                    "sam3_box_xyxy": sam3_box_xyxy,
                    "best_iou": float(best_iou),
                    "matched_iou": float(matched_pair[1]) if matched_pair is not None else 0.0,
                    "matched": matched_pair is not None,
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
        "num_sam_boxes": sum(len(v) for v in sam_by_image.values()),
        "num_loc_boxes": len(loc_boxes),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "sam3_unique_tasks": int(len(task_payloads)),
        "sam3_workers_used": int(min(len(worker_devices), len(task_payloads))),
        "sam3_shard_metrics": dict(sam3_shard_metrics),
        "prediction_diagnostics": diagnostics,
    }


def compute_score_batched(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Sequence[Any],
    raw_prompts: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    del data_sources, ground_truths
    total = len(solution_strs)
    if raw_prompts is None:
        raw_prompts_list = [None] * total
    elif isinstance(raw_prompts, np.ndarray):
        raw_prompts_list = raw_prompts.tolist()
    else:
        raw_prompts_list = list(raw_prompts)
    if len(raw_prompts_list) < total:
        raw_prompts_list.extend([None] * (total - len(raw_prompts_list)))

    reward_config = _build_reward_config(kwargs)
    prompt_template = _load_prompt_template(kwargs.get("gemini_answer_grader_prompt_path"))
    format_valid_flags = _build_any_supported_format_valid_flags(solution_strs, total)

    return_score_dict = _coerce_bool(kwargs.get("gemini_return_score_dict", True), True)
    requests_by_key: dict[str, dict[str, Any]] = {}
    key_to_indices: dict[str, list[int]] = {}
    results_payloads: list[dict[str, Any]] = [
        {
            "score": 0.0,
            "verdict": "incorrect",
            "reason": "",
            "failed": True,
            "error": "not_scored",
            "raw_text": "",
        }
        for _ in range(total)
    ]

    for idx in range(total):
        solution_str = str(solution_strs[idx] or "")
        if not format_valid_flags[idx]:
            results_payloads[idx] = {
                "score": 0.0,
                "verdict": "incorrect",
                "reason": "Invalid output format: expected exactly one non-empty <reason> block and one non-empty <answer> block.",
                "failed": False,
                "error": "invalid_format",
                "raw_text": "",
            }
            continue
        raw_prompt = raw_prompts_list[idx]
        query = _extract_query_text_from_raw_prompt(raw_prompt)
        originals = _extract_original_images_from_raw_prompt(raw_prompt)
        images = [image for image, _ in originals]
        image_fingerprints = [fingerprint for _, fingerprint in originals]
        request_key = _request_cache_key(
            query=query,
            candidate_response=solution_str,
            image_fingerprints=image_fingerprints,
            model=reward_config["model"],
        )
        key_to_indices.setdefault(request_key, []).append(idx)
        if request_key in requests_by_key:
            continue
        requests_by_key[request_key] = {
            "query": query,
            "candidate_response": solution_str,
            "images": images,
            "cache_key": request_key,
        }

    uncached_tasks: list[dict[str, Any]] = []
    with _RESPONSE_CACHE_LOCK:
        for cache_key, payload in requests_by_key.items():
            cached = _RESPONSE_CACHE.get(cache_key)
            if cached is not None:
                normalized_cached = _normalize_response_dict(cached if isinstance(cached, dict) else {})
                normalized_cached["failed"] = bool(cached.get("failed", False)) if isinstance(cached, dict) else False
                normalized_cached["error"] = str(cached.get("error", "") or "") if isinstance(cached, dict) else ""
                normalized_cached["raw_text"] = str(cached.get("raw_text", "") or "") if isinstance(cached, dict) else ""
                for idx in key_to_indices.get(cache_key, []):
                    results_payloads[idx] = dict(normalized_cached)
            else:
                uncached_tasks.append(payload)

    if uncached_tasks:
        max_workers = int(reward_config["max_workers"])
        if max_workers <= 1:
            for task in uncached_tasks:
                cache_key = task["cache_key"]
                try:
                    judgment = _generate_gemini_judgment(
                        config=reward_config,
                        prompt_template=prompt_template,
                        query=task["query"],
                        candidate_response=task["candidate_response"],
                        images=task["images"],
                    )
                except Exception as exc:
                    _gemini_log(
                        f"unhandled sequential Gemini reward exception; assigning reward 0. error={exc}"
                    )
                    judgment = {
                        "score": 0.0,
                        "verdict": "incorrect",
                        "reason": "",
                        "raw_text": "",
                        "failed": True,
                        "error": str(exc),
                    }
                normalized = _normalize_response_dict(judgment)
                normalized["failed"] = bool(judgment.get("failed", False))
                normalized["error"] = str(judgment.get("error", "") or "")
                normalized["raw_text"] = str(judgment.get("raw_text", "") or "")
                with _RESPONSE_CACHE_LOCK:
                    _RESPONSE_CACHE[cache_key] = dict(normalized)
                for idx in key_to_indices.get(cache_key, []):
                    results_payloads[idx] = dict(normalized)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(
                        _generate_gemini_judgment,
                        config=reward_config,
                        prompt_template=prompt_template,
                        query=task["query"],
                        candidate_response=task["candidate_response"],
                        images=task["images"],
                    ): task["cache_key"]
                    for task in uncached_tasks
                }
                for future in as_completed(future_to_key):
                    cache_key = future_to_key[future]
                    try:
                        judgment = future.result()
                    except Exception as exc:
                        _gemini_log(f"unhandled future exception; assigning reward 0. error={exc}")
                        judgment = {
                            "score": 0.0,
                            "verdict": "incorrect",
                            "reason": "",
                            "raw_text": "",
                            "failed": True,
                            "error": str(exc),
                        }
                    normalized = _normalize_response_dict(judgment)
                    normalized["failed"] = bool(judgment.get("failed", False))
                    normalized["error"] = str(judgment.get("error", "") or "")
                    normalized["raw_text"] = str(judgment.get("raw_text", "") or "")
                    with _RESPONSE_CACHE_LOCK:
                        _RESPONSE_CACHE[cache_key] = dict(normalized)
                    for idx in key_to_indices.get(cache_key, []):
                        results_payloads[idx] = dict(normalized)

    if return_score_dict:
        return [
            {
                "score": float(_clip01(payload.get("score", 0.0))),
                "gemini_verdict": str(payload.get("verdict", "") or ""),
                "gemini_reason": str(payload.get("reason", "") or ""),
                "gemini_failed": bool(payload.get("failed", False)),
                "gemini_error": str(payload.get("error", "") or ""),
                "gemini_raw_text": str(payload.get("raw_text", "") or ""),
            }
            for payload in results_payloads
        ]

    return [float(_clip01(payload.get("score", 0.0))) for payload in results_payloads]


def compute_loc_call_rewards_batched(
    data_sources: Sequence[str],
    loc_call_records: Sequence[Sequence[dict]] | Sequence[Any],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Sequence[Any],
    uids: Sequence[str] | Sequence[str] | None = None,
    loc_verifier_call_records: Sequence[Sequence[dict]] | Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[list[float]]:
    """Gemini-path detection objective using relevant_objects as SAM3 targets."""
    del data_sources, ground_truths, loc_verifier_call_records, uids
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
    sam3_shard_enable = _coerce_bool(
        kwargs.get("genver_sam3_shard_enable", sam3_cache_shard_enable),
        sam3_cache_shard_enable,
    )

    loc_call_records_list: list[list[dict[str, Any]]] = []
    for sample_calls in loc_call_records:
        if isinstance(sample_calls, np.ndarray):
            sample_calls = sample_calls.tolist()
        if not isinstance(sample_calls, (list, tuple)):
            loc_call_records_list.append([])
            continue
        loc_call_records_list.append([record for record in sample_calls if isinstance(record, dict)])

    num_samples = len(loc_call_records_list)
    if isinstance(extra_infos, np.ndarray):
        extra_infos_list = extra_infos.tolist()
    elif isinstance(extra_infos, (list, tuple)):
        extra_infos_list = list(extra_infos)
    else:
        extra_infos_list = [None] * num_samples
    if len(extra_infos_list) < num_samples:
        extra_infos_list.extend([None] * (num_samples - len(extra_infos_list)))

    raw_prompts = kwargs.get("raw_prompts")
    if isinstance(raw_prompts, np.ndarray):
        raw_prompts_list = raw_prompts.tolist()
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

        raw_prompt = raw_prompts_list[sample_idx]
        originals = _extract_original_images_from_raw_prompt(raw_prompt)
        relevant_objects = _extract_relevant_objects(extra_infos_list[sample_idx])
        eval_cache: dict[str, dict[str, Any]] = {}
        pq_values: list[float] = []

        for call_record in call_records:
            entries = _extract_bbox_2d_entries_from_call_record(call_record)
            if sam3_cache_shard_enable:
                dedupe_key = _serialize_entries_and_targets_for_dedupe(entries, relevant_objects)
                eval_result = eval_cache.get(dedupe_key)
                if eval_result is None:
                    eval_result = _evaluate_loc_entries_with_relevant_objects(
                        entries=entries,
                        originals=originals,
                        relevant_objects=relevant_objects,
                        genver_sam3_confidence_threshold=sam3_conf_threshold,
                        genver_sam3_device=sam3_device,
                        genver_sam3_devices=sam3_devices,
                        genver_sam3_shard_workers=sam3_shard_workers,
                        genver_sam3_shard_enable=sam3_shard_enable,
                        genver_sam3_checkpoint_path=sam3_checkpoint_path,
                        genver_sam3_load_from_hf=sam3_load_from_hf,
                    )
                    eval_cache[dedupe_key] = eval_result
                eval_result = dict(eval_result)
            else:
                eval_result = _evaluate_loc_entries_with_relevant_objects(
                    entries=entries,
                    originals=originals,
                    relevant_objects=relevant_objects,
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
        for idx, call_record in enumerate(call_records):
            quality = _clip01(pq_values[idx] if idx < len(pq_values) else 0.0)
            call_index = _safe_int(call_record.get("call_index", idx), idx)
            call_indices.append(call_index)
            call_record["genver_loc_reward_quality"] = float(quality)
            call_record["genver_loc_reward_raw"] = 0.0

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
    loc_verifier_call_records: Sequence[dict] | Sequence[Any],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Sequence[Any],
    parent_row_indices: Sequence[int] | Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Gemini-path grounding verifier reward; loc eval source comes from relevant_objects-based PQ."""
    del data_sources, ground_truths, extra_infos
    eps = 1e-8
    loc_eval_lookup_raw = kwargs.get("_genver_loc_eval_lookup", {})
    loc_eval_lookup = loc_eval_lookup_raw if isinstance(loc_eval_lookup_raw, dict) else {}

    call_records: list[dict[str, Any]] = []
    for record in loc_verifier_call_records:
        call_records.append(record if isinstance(record, dict) else {})

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

        parent_row_index = _safe_int(parent_row_index_list[idx] if idx < len(parent_row_index_list) else idx, idx)
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
                prediction_diagnostics = [pred for pred in raw_prediction_diagnostics if isinstance(pred, dict)]
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
        parent_row_index = _safe_int(parent_row_index_list[idx] if idx < len(parent_row_index_list) else idx, idx)
        parent_to_indices.setdefault(int(parent_row_index), []).append(idx)

    for parent_row_index, parent_indices in parent_to_indices.items():
        if len(parent_indices) == 0:
            continue

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
            if isinstance(loc_eval, dict):
                parent_call_pq[call_index] = _clip01(loc_eval.get("pq", 0.0))

        for idx in parent_indices:
            call_index = _safe_int(call_records[idx].get("call_index", idx), idx)
            if call_index not in parent_call_pq and eval_results[idx] is not None:
                parent_call_pq[call_index] = float(eval_results[idx].get("pq", 0.0))

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
            raw_reward = float(usefulness)

            call_records[idx]["genver_loc_verifier_delta_pq"] = float(delta)
            call_records[idx]["genver_loc_verifier_usefulness"] = float(usefulness)
            call_records[idx]["genver_loc_verifier_reward_raw"] = float(raw_reward)
            call_records[idx]["genver_loc_verifier_valid_for_reward"] = True
            raw_rewards[idx] = float(raw_reward)
            valid_indices.append(idx)

        valid_count = len(valid_indices)
        if valid_count <= 0:
            continue
        scale = 1.0 / float(valid_count)
        for idx in valid_indices:
            reward = float(raw_rewards[idx]) * scale
            call_records[idx]["genver_loc_verifier_reward"] = reward
            rewards[idx] = reward

    return rewards


def compute_answer_logic_verifier_rewards_batched(
    data_sources: Sequence[str],
    answer_logic_verifier_call_records: Sequence[dict] | Sequence[Any],
    answer_call_records: Sequence[Sequence[dict]] | Sequence[Any],
    ground_truths: Sequence[str],
    extra_infos: Sequence[dict] | Sequence[Any],
    parent_row_indices: Sequence[int] | Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    del data_sources, ground_truths, extra_infos, parent_row_indices, kwargs
    eps = 1e-6
    if isinstance(answer_call_records, np.ndarray):
        answer_call_records_list = answer_call_records.tolist()
    elif isinstance(answer_call_records, (list, tuple)):
        answer_call_records_list = list(answer_call_records)
    else:
        answer_call_records_list = []
    if len(answer_call_records_list) < len(answer_logic_verifier_call_records):
        answer_call_records_list.extend(
            [[] for _ in range(len(answer_logic_verifier_call_records) - len(answer_call_records_list))]
        )
    format_valid_flags: list[bool] = []
    for idx in range(len(answer_logic_verifier_call_records)):
        solution = _extract_final_answer_solution_from_call_records(answer_call_records_list[idx])
        format_valid_flags.append(_is_any_supported_genver_format(solution))

    rewards: list[float] = []
    for idx, record in enumerate(answer_logic_verifier_call_records):
        if not isinstance(record, dict):
            rewards.append(0.0)
            continue

        if not format_valid_flags[idx]:
            record["genver_answer_logic_verifier_parse_valid"] = False
            record["genver_answer_logic_verifier_feedback_valid"] = False
            record["genver_answer_logic_verifier_format_valid"] = False
            record["genver_answer_logic_verifier_usefulness"] = 0.0
            record["genver_answer_logic_verifier_reward_raw"] = 0.0
            record["genver_answer_logic_verifier_reward"] = 0.0
            record["genver_answer_logic_verifier_valid_for_reward"] = False
            rewards.append(0.0)
            continue

        parse_valid = bool(record.get("logic_feedback_parse_valid", False))
        feedback_valid = bool(record.get("logic_feedback_valid_for_reward", False))
        edit_source = str(record.get("logic_edit_source", "") or "").strip().lower()
        current_answer_score = _clip01(record.get("genver_answer_logic_verifier_current_answer_score", 0.0))
        final_answer_score = _clip01(record.get("genver_answer_logic_verifier_final_answer_score", 0.0))
        self_edit_score = _clip01(record.get("genver_answer_logic_verifier_self_edit_score", 0.0))

        record["genver_answer_logic_verifier_current_answer_score"] = float(current_answer_score)
        record["genver_answer_logic_verifier_final_answer_score"] = float(final_answer_score)
        record["genver_answer_logic_verifier_self_edit_score"] = float(self_edit_score)
        record["genver_answer_logic_verifier_parse_valid"] = bool(parse_valid)
        record["genver_answer_logic_verifier_feedback_valid"] = bool(feedback_valid)
        record["genver_answer_logic_verifier_format_valid"] = True

        if not parse_valid or not feedback_valid:
            record["genver_answer_logic_verifier_usefulness"] = 0.0
            record["genver_answer_logic_verifier_reward_raw"] = 0.0
            record["genver_answer_logic_verifier_reward"] = 0.0
            record["genver_answer_logic_verifier_valid_for_reward"] = False
            rewards.append(0.0)
            continue

        delta = float(final_answer_score - current_answer_score)
        if delta > eps:
            usefulness = 1.0
        elif delta < -eps:
            usefulness = 0.0
        elif current_answer_score > DEFAULT_STRONG_ANSWER_THRESHOLD:
            usefulness = 1.0
        else:
            usefulness = 0.3

        if edit_source == "teacher":
            reward_raw = float(self_edit_score)
        else:
            reward_raw = float(0.7 * self_edit_score + 0.3 * usefulness)

        reward = _clip01(reward_raw)
        record["genver_answer_logic_verifier_delta_answer_reward"] = float(delta)
        record["genver_answer_logic_verifier_usefulness"] = float(usefulness)
        record["genver_answer_logic_verifier_reward_raw"] = float(reward_raw)
        record["genver_answer_logic_verifier_reward"] = float(reward)
        record["genver_answer_logic_verifier_valid_for_reward"] = True
        rewards.append(float(reward))
    return rewards


__all__ = [
    "DEFAULT_STRONG_ANSWER_THRESHOLD",
    "build_gemini_runtime_config",
    "load_gemini_prompt_template",
    "generate_gemini_answer_judgment",
    "generate_gemini_logic_teacher_judgment",
    "compute_score_batched",
    "compute_loc_call_rewards_batched",
    "compute_loc_verifier_rewards_batched",
    "compute_answer_logic_verifier_rewards_batched",
]

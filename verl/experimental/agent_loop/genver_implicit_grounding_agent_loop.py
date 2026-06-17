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

import hashlib
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.genver_gemini_objective_agent_loop import (
    GenVerGeminiObjectiveAgentLoop,
)
from training.gemini_objectives import load_gemini_prompt_template


@register("genver_implicit_grounding_agent")
class GenVerImplicitGroundingAgentLoop(GenVerGeminiObjectiveAgentLoop):
    """Implicit-grounding Gemini loop.

    Pipeline:
    query -> <reason><answer> -> logic self-verifier call(s) -> rewritten <reason><answer>.

    No explicit grounding/bbox generation, no grounding verifier, no grounding context compaction.
    """

    def __init__(
        self,
        *args: Any,
        logic_verifier_rounds: int = 1,
        logic_verifier_rounds_min: Optional[int] = None,
        logic_verifier_rounds_max: Optional[int] = None,
        logic_verifier_max_new_tokens: int = 96,
        logic_self_verifier_prompt_path: Optional[str] = None,
        gemini_logic_teacher_prompt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            loc_verifier_rounds=0,
            logic_verifier_rounds=1,
            logic_verifier_max_new_tokens=logic_verifier_max_new_tokens,
            logic_self_verifier_prompt_path=logic_self_verifier_prompt_path,
            gemini_logic_teacher_prompt_path=gemini_logic_teacher_prompt_path,
            **kwargs,
        )

        prompts_dir = self._resolve_training_prompts_dir()
        if logic_self_verifier_prompt_path:
            prompt_path = Path(logic_self_verifier_prompt_path)
        else:
            prompt_path = prompts_dir / "logic_self_verifier_gemini_implicit_grounding_instructions.txt"
        self.logic_self_verifier_template = self._read_prompt_file(prompt_path)

        self.gemini_logic_teacher_prompt = load_gemini_prompt_template(
            gemini_logic_teacher_prompt_path,
            default_filename="gemini_logic_teacher_judge_implicit_grounding_instructions.txt",
        )
        fixed_rounds = max(0, int(logic_verifier_rounds))
        rounds_min = fixed_rounds if logic_verifier_rounds_min is None else max(0, int(logic_verifier_rounds_min))
        rounds_max = fixed_rounds if logic_verifier_rounds_max is None else max(0, int(logic_verifier_rounds_max))
        if rounds_max < rounds_min:
            rounds_max = rounds_min
        self.logic_verifier_rounds = fixed_rounds
        self.logic_verifier_rounds_min = int(rounds_min)
        self.logic_verifier_rounds_max = int(rounds_max)
        self.total_epochs = int(
            getattr(self.config.trainer, "total_epochs", 1) or 1
        )
        if self.total_training_steps > 0 and self.total_epochs > 0:
            self.steps_per_epoch = max(1, self.total_training_steps // self.total_epochs)
        else:
            self.steps_per_epoch = 1

    def _sample_logic_verifier_rounds(
        self,
        *,
        uid: str,
        global_steps: int,
    ) -> int:
        rounds_min = max(0, int(self.logic_verifier_rounds_min))
        rounds_max = max(rounds_min, int(self.logic_verifier_rounds_max))
        if rounds_max <= rounds_min:
            return rounds_min
        digest = hashlib.sha1(
            f"logic_rounds:{uid}:{global_steps}".encode("utf-8")
        ).digest()
        draw = int.from_bytes(digest[:8], byteorder="big")
        span = rounds_max - rounds_min + 1
        return int(rounds_min + (draw % span))

    def _choose_logic_edit_source(
        self,
        *,
        uid: str,
        global_steps: int,
        validate_mode: bool,
    ) -> str:
        if validate_mode:
            return "self"

        # Ramp p_self only during epoch 1, then force self for later epochs.
        # step index 0..steps_per_epoch-1 => p_self in (0, 1], step >= steps_per_epoch => 1.0
        if global_steps >= self.steps_per_epoch:
            p_self = 1.0
        else:
            progress_first_epoch = float(max(0, global_steps) + 1) / float(self.steps_per_epoch)
            p_self = max(0.0, min(1.0, progress_first_epoch))

        digest = hashlib.sha1(f"{uid}:{global_steps}".encode("utf-8")).digest()
        threshold = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
        return "self" if threshold < p_self else "teacher"

    def _build_initial_answer_prompt(self, query: str) -> str:
        query_text = query.strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags with numbered steps "
            "(1., 2., 3., ... one step per line), then putting ONLY your final answer inside <answer>. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "For counting questions, count only clearly visible instances once; do not infer hidden or distant instances. "
            "Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )

    def _build_logic_self_verifier_prompt(self, query: str, latest_answer_output: str) -> str:
        return self.logic_self_verifier_template.format(
            query=query.strip(),
            answer=str(latest_answer_output or "").strip(),
        )

    async def _generate_answer_from_prompt(
        self,
        *,
        images: list[Image.Image],
        prompt_text: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[str, str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, Any]]:
        messages = self._build_messages(prompt_text, images)
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
            prompt_text,
            output_text,
            prompt_ids,
            output_ids,
            output_log_probs,
            serialized_images,
            answer_multi_modal_inputs,
        )

    async def _build_answer_rewrite_aux_messages(
        self,
        *,
        images: list[Image.Image],
        query: str,
        current_answer_output: str,
        logic_feedback: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[str, str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, Any]]:
        rewrite_prompt = (
            f"{query.strip()}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            "I have some feedback for you to incorporate. "
            "Please update the <reason> using the feedback, revising the referenced numbered reasoning steps only, "
            "then output a final <answer> that follows from the updated reasoning.\n"
            f"Feedback:\n{logic_feedback}\n"
            "You MUST update the reasoning to incorporate the feedback. "
            "You MUST keep the <reason> step-indexed with numbered lines (1., 2., 3., ... one step per line). "
            "You MUST then produce the final answer from that updated reasoning. "
            "You MUST incorporate the feedback and MUST NOT make unrelated changes. "
            "Apply the feedback silently: do not mention feedback, instructions, or edits in <reason>. "
            "Keep <reason> concise and non-repetitive. "
            "Please output exactly one full <reason> block and then one full <answer> block. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "For counting questions, count only clearly visible instances once; do not infer hidden or distant instances. "
            "Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else."
        )
        return await self._generate_answer_from_prompt(
            images=images,
            prompt_text=rewrite_prompt,
            sampling_params=sampling_params,
            metrics=metrics,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        validate_mode = bool(kwargs.get("validate", False))
        uid = str(kwargs.get("uid", "") or "")
        raw_global_steps = kwargs.get("global_steps", -1)
        global_steps = int(raw_global_steps if raw_global_steps is not None else -1)

        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        raw_images_rgb = [img.convert("RGB") for img in raw_images]
        images = [self._resize_longest_side(img, self.max_image_side) for img in raw_images_rgb]
        gemini_images = raw_images_rgb if raw_images_rgb else images

        metrics: dict[str, Any] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "num_preempted": -1,
        }

        # Initial answer call (no explicit grounding).
        (
            answer_prompt_text,
            answer_output_text,
            answer_prompt_token_ids,
            answer_output_token_ids,
            answer_output_log_probs,
            answer_images,
            answer_multi_modal_inputs,
        ) = await self._generate_answer_from_prompt(
            images=images,
            prompt_text=self._build_initial_answer_prompt(query),
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
            "answer_latest_bbox_block": "",
            "answer_solution_str": current_answer_output,
        }

        genver_answer_calls: list[dict[str, Any]] = [
            {
                "call_index": int(answer_call_index),
                "answer_prompt_text": answer_prompt_text,
                "answer_output_text": current_answer_output,
                "answer_solution_str": current_answer_aux_record["answer_solution_str"],
            }
        ]
        genver_answer_logic_verifier_calls: list[dict[str, Any]] = []

        sampled_logic_rounds = self._sample_logic_verifier_rounds(
            uid=uid,
            global_steps=global_steps,
        )
        selected_edit_source_for_sample = self._choose_logic_edit_source(
            uid=uid,
            global_steps=global_steps,
            validate_mode=validate_mode,
        )
        final_answer_call = dict(current_answer_aux_record)

        for logic_round_index in range(sampled_logic_rounds):
            logic_prompt_text = self._build_logic_self_verifier_prompt(query, current_answer_output)
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
            self_feedback, self_parse_valid, self_feedback_info = self._parse_logic_step_edits_optional(
                logic_output_text, current_answer_output
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
                    latest_bbox_block="",
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
                teacher_feedback, teacher_parse_valid, teacher_feedback_info = self._parse_logic_step_edits_optional(
                    teacher_output_text, current_answer_output
                )
                current_answer_score = float(teacher_judgment.get("current_answer_score", 0.0) or 0.0)
                self_edit_score = float(teacher_judgment.get("self_edit_score", 0.0) or 0.0)
                self_edit_reason = str(teacher_judgment.get("self_edit_reason", "") or "").strip()

            teacher_judgment_failed = bool(teacher_judgment.get("failed", False))
            edit_source = str(selected_edit_source_for_sample)
            force_logic_reward_invalid = bool(
                (not validate_mode) and teacher_judgment_failed and (edit_source == "teacher")
            )

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
                "round_index": int(logic_round_index),
                "sampled_logic_rounds": int(sampled_logic_rounds),
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
                "logic_teacher_output_raw_text": str(teacher_judgment.get("raw_text", "") or ""),
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
                "logic_edit_source_sample": str(selected_edit_source_for_sample),
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

            final_answer_score = float(current_answer_score)
            rewrite_skipped_no_edits = not bool(str(selected_feedback or "").strip())
            if rewrite_skipped_no_edits:
                final_answer_call = dict(current_answer_aux_record)
                if validate_mode:
                    final_answer_score = 0.0
                    final_answer_call["answer_gemini_score_time_s"] = 0.0
                elif not teacher_judgment_failed:
                    final_answer_score = float(current_answer_score)
                    final_answer_call["answer_reward_override"] = float(current_answer_score)
                    final_answer_call["answer_gemini_score_time_s"] = 0.0
                else:
                    t_answer_grade_start = time.perf_counter()
                    final_answer_judgment = await self._request_gemini_answer_score(
                        query=query,
                        candidate_response=final_answer_call["answer_solution_str"],
                        images=gemini_images,
                    )
                    answer_gemini_score_time_s = float(time.perf_counter() - t_answer_grade_start)
                    final_answer_score = float(final_answer_judgment.get("score", 0.0) or 0.0)
                    final_answer_call["answer_reward_override"] = float(final_answer_score)
                    final_answer_call["answer_gemini_score_time_s"] = float(answer_gemini_score_time_s)
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
                    "answer_latest_bbox_block": "",
                    "answer_solution_str": current_answer_output,
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
                    answer_gemini_score_time_s = float(time.perf_counter() - t_answer_grade_start)
                    final_answer_score = float(final_answer_judgment.get("score", 0.0) or 0.0)
                    final_answer_call["answer_reward_override"] = float(final_answer_score)
                    final_answer_call["answer_gemini_score_time_s"] = float(answer_gemini_score_time_s)
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
                logic_teacher_time_s + float(final_answer_call.get("answer_gemini_score_time_s", 0.0) or 0.0)
            )
            logic_call_record["genver_answer_logic_verifier_final_answer_score"] = float(final_answer_score)
            logic_call_record["genver_answer_logic_verifier_rewrite_skipped_no_edits"] = bool(rewrite_skipped_no_edits)
            logic_call_record["genver_answer_logic_verifier_logic_teacher_time_s"] = float(logic_teacher_time_s)
            logic_call_record["genver_answer_logic_verifier_answer_gemini_score_time_s"] = float(
                final_answer_call.get("answer_gemini_score_time_s", 0.0) or 0.0
            )
            logic_call_record["genver_answer_logic_verifier_gemini_total_time_s"] = float(
                final_answer_call.get("gemini_total_time_s", 0.0) or 0.0
            )
            genver_answer_logic_verifier_calls.append(logic_call_record)
            current_answer_aux_record = dict(final_answer_call)
            current_answer_output = str(final_answer_call.get("answer_output_text", "") or "")

        genver_answer_aux_call = dict(final_answer_call)

        # Keep main trajectory as the final rewritten answer only.
        response_ids = list(final_answer_call.get("answer_output_token_ids", []) or [])[: self.response_length]
        response_mask = [1] * len(response_ids)
        final_output_log_probs_raw = list(final_answer_call.get("answer_output_log_probs", []) or [])
        final_output_log_probs = [
            float(x) if isinstance(x, (int, float)) else 0.0 for x in final_output_log_probs_raw
        ][: len(response_ids)]
        response_logprobs: Optional[list[float]]
        if sampling_params.get("logprobs"):
            if len(final_output_log_probs) < len(response_ids):
                final_output_log_probs.extend([0.0] * (len(response_ids) - len(final_output_log_probs)))
            response_logprobs = final_output_log_probs
        else:
            response_logprobs = None

        genver_answer_mask = [1] * len(response_ids)
        genver_loc_mask = [0] * len(response_ids)
        final_prompt_ids = list(final_answer_call.get("answer_prompt_token_ids", []) or [])

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={"images": images} if images else {},
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "genver_answer_mask": genver_answer_mask,
                "genver_loc_mask": genver_loc_mask,
                "genver_loc_calls": [],
                "genver_loc_verifier_calls": [],
                "genver_answer_aux_call": genver_answer_aux_call,
                "genver_answer_calls": genver_answer_calls,
                "genver_answer_logic_verifier_calls": genver_answer_logic_verifier_calls,
            },
        )

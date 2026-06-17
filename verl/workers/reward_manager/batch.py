# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        try:
            response_mask = data.batch["response_mask"]
        except Exception:
            response_mask = None

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        answer_aux_calls_raw = data.non_tensor_batch.get("genver_answer_aux_call", None)
        if answer_aux_calls_raw is not None and hasattr(answer_aux_calls_raw, "tolist"):
            answer_aux_calls_raw = answer_aux_calls_raw.tolist()
        if not isinstance(answer_aux_calls_raw, (list, tuple)):
            answer_aux_calls_raw = [None] * len(data)

        responses_str = []
        for i in range(len(data)):
            if response_mask is not None and tuple(response_mask.shape) == tuple(response_ids.shape):
                valid_response_ids = response_ids[i][response_mask[i] > 0]
            else:
                valid_len = valid_response_lengths[i]
                valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            if i < len(answer_aux_calls_raw):
                call_record = answer_aux_calls_raw[i]
                # Be permissive with object-array nesting so aux answer override
                # works across different DataProto packing paths.
                if isinstance(call_record, np.ndarray):
                    if call_record.shape == ():
                        call_record = call_record.item()
                    elif call_record.size == 1:
                        call_record = call_record.reshape(-1)[0]
                if isinstance(call_record, list) and len(call_record) == 1 and isinstance(call_record[0], dict):
                    call_record = call_record[0]
                if isinstance(call_record, dict):
                    aux_solution = str(call_record.get("answer_solution_str", "") or "").strip()
                    if aux_solution:
                        response_str = aux_solution
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        raw_prompts = data.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is not None and hasattr(raw_prompts, "tolist"):
            raw_prompts = raw_prompts.tolist()

        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        score_kwargs: dict[str, Any] = {
            "data_sources": data_sources,
            "solution_strs": responses_str,
            "ground_truths": ground_truths,
            "extra_infos": extras,
            **self.reward_kwargs,
        }
        if raw_prompts is not None:
            score_kwargs["raw_prompts"] = raw_prompts

        try:
            scores = self.compute_score(**score_kwargs)
        except TypeError as exc:
            # Backward compatibility: older reward functions may not accept raw_prompts.
            if "raw_prompts" in str(exc) and "raw_prompts" in score_kwargs:
                score_kwargs.pop("raw_prompts", None)
                scores = self.compute_score(**score_kwargs)
            else:
                raise

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        try:
            response_mask = data.batch["response_mask"]
        except Exception:
            response_mask = None
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                if response_mask is not None and tuple(response_mask.shape) == tuple(response_ids.shape):
                    response_str = self.tokenizer.decode(
                        data.batch["responses"][i][response_mask[i] > 0],
                        skip_special_tokens=True,
                    )
                else:
                    response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

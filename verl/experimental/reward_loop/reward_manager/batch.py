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

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """Reward manager that calls a batched reward function for a single sample."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        try:
            response_mask = data_item.batch["response_mask"]
        except Exception:
            response_mask = None
        if response_mask is not None and tuple(response_mask.shape) == tuple(response_ids.shape):
            valid_response_ids = response_ids[response_mask > 0]
        else:
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        payload = dict(
            data_sources=[data_source],
            solution_strs=[response_str],
            ground_truths=[ground_truth],
            extra_infos=[extra_info],
            **extra_reward_kwargs,
        )

        if self.is_async_reward_score:
            result = await self.compute_score(**payload)
        else:
            result = await self.loop.run_in_executor(None, lambda: self.compute_score(**payload))

        if isinstance(result, list):
            result_item = result[0] if result else 0.0
        else:
            result_item = result

        reward_extra_info = {}
        if isinstance(result_item, dict):
            reward = result_item.get("score", 0.0)
            reward_extra_info.update(result_item)
        else:
            reward = result_item
            reward_extra_info["acc"] = reward

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

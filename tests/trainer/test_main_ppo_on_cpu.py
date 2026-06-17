# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest

from omegaconf import OmegaConf

from verl.trainer.main_ppo import _get_hf_trust_remote_code


class TestMainPPOTrustRemoteCode(unittest.TestCase):
    def test_prefers_model_trust_remote_code_for_multimodal_models(self) -> None:
        config = OmegaConf.create(
            {
                "data": {"trust_remote_code": False},
                "actor_rollout_ref": {"model": {"trust_remote_code": True}},
            }
        )

        self.assertTrue(_get_hf_trust_remote_code(config))

    def test_keeps_data_trust_remote_code_when_enabled(self) -> None:
        config = OmegaConf.create(
            {
                "data": {"trust_remote_code": True},
                "actor_rollout_ref": {"model": {"trust_remote_code": False}},
            }
        )

        self.assertTrue(_get_hf_trust_remote_code(config))

    def test_returns_false_when_both_flags_are_disabled(self) -> None:
        config = OmegaConf.create(
            {
                "data": {"trust_remote_code": False},
                "actor_rollout_ref": {"model": {"trust_remote_code": False}},
            }
        )

        self.assertFalse(_get_hf_trust_remote_code(config))


if __name__ == "__main__":
    unittest.main()

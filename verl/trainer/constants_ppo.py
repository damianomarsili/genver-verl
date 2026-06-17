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

import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    },
}


def get_default_ray_temp_dir(
    repo_root: str | os.PathLike[str] | None = None,
    hostname: str | None = None,
) -> str:
    """Return the default Ray temp dir on the writable repo volume.

    Ray defaults to `/tmp`, but this repo often runs on machines where `/tmp`
    is much smaller than the workspace volume. Use a repo-local path instead,
    split per host so multi-node jobs do not collide on shared storage.
    """
    base_dir = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    node_name = (hostname or socket.gethostname() or "localhost").split(".", maxsplit=1)[0]
    return os.fspath(base_dir / ".ray" / node_name)


def with_default_ray_init_kwargs(
    ray_init_kwargs: Mapping[str, Any] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Fill Ray init kwargs with a repo-local temp dir when none is configured."""
    resolved_kwargs = dict(ray_init_kwargs or {})

    configured_temp_dir = resolved_kwargs.get("_temp_dir")
    if configured_temp_dir:
        resolved_kwargs["_temp_dir"] = os.fspath(configured_temp_dir)
        return resolved_kwargs

    env_temp_dir = str(os.environ.get("RAY_TMPDIR", "") or "").strip()
    if env_temp_dir:
        resolved_kwargs["_temp_dir"] = env_temp_dir
        return resolved_kwargs

    default_temp_dir = get_default_ray_temp_dir(repo_root=repo_root, hostname=hostname)
    Path(default_temp_dir).mkdir(parents=True, exist_ok=True)
    resolved_kwargs["_temp_dir"] = default_temp_dir
    return resolved_kwargs


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": working_dir} if isinstance(working_dir, str) and working_dir else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env

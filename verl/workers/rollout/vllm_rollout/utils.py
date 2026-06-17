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
import ctypes
import gc
import json
import logging
import os
import platform
import signal
import tempfile
import threading
from multiprocessing import shared_memory
from types import MethodType
from typing import Any, Callable, TypedDict, get_args

import torch
import zmq

from verl.utils.device import get_torch_device, is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

MOLMO2_EMBEDDING_KEY = "model.transformer.wte.embedding"
MOLMO2_NEW_EMBEDDING_KEY = "model.transformer.wte.new_embedding"
MOLMO2_MERGED_EMBEDDING_KEY_SUFFIXES = (
    "language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
        return "NPU-" + npu_visible_devices[device_id]
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    from vllm.config.lora import MaxLoRARanks

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


def _is_molmo2_vllm_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "molmo2":
        return True

    text_config = getattr(config, "text_config", None)
    if getattr(text_config, "model_type", None) == "molmo2_text":
        return True

    return type(model).__name__ == "Molmo2ForConditionalGeneration"


def _get_molmo2_base_vocab_size(model: Any) -> int | None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None) or config
    return getattr(text_config, "vocab_size", None)


def _split_molmo2_merged_embedding(
    weights: list[tuple[str, torch.Tensor]],
    model: Any,
) -> list[tuple[str, torch.Tensor]]:
    """Convert merged Molmo2 embeddings into the split keys vLLM expects."""
    base_vocab_size = _get_molmo2_base_vocab_size(model)
    if base_vocab_size is None:
        return weights

    normalized_weights: list[tuple[str, torch.Tensor]] = []
    for name, weight in weights:
        if any(name.endswith(suffix) for suffix in MOLMO2_MERGED_EMBEDDING_KEY_SUFFIXES):
            extra_rows = weight.shape[0] - base_vocab_size
            if extra_rows <= 0:
                logger.warning(
                    "Molmo2 merged embedding %s has incompatible shape %s for base vocab size %s; "
                    "loading it unchanged.",
                    name,
                    tuple(weight.shape),
                    base_vocab_size,
                )
                normalized_weights.append((name, weight))
                continue

            normalized_weights.append((MOLMO2_EMBEDDING_KEY, weight[:base_vocab_size]))
            normalized_weights.append((MOLMO2_NEW_EMBEDDING_KEY, weight[base_vocab_size:]))
        else:
            normalized_weights.append((name, weight))

    return normalized_weights


def _prepare_molmo2_bucket_weights(
    model: Any,
    weights: list[tuple[str, torch.Tensor]],
    embedding_cache: dict[str, torch.Tensor],
    pending_weights: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]] | None, dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
    """Make bucketed Molmo2 updates compatible with vLLM's full-call embedding merge."""
    normalized_weights = _split_molmo2_merged_embedding(weights, model)

    updated_cache = dict(embedding_cache)
    next_pending = list(pending_weights)
    for name, weight in normalized_weights:
        if "wte.embedding" in name:
            updated_cache["embedding"] = weight
        elif "wte.new_embedding" in name:
            updated_cache["new_embedding"] = weight
        else:
            next_pending.append((name, weight))

    if "embedding" not in updated_cache or "new_embedding" not in updated_cache:
        return None, updated_cache, next_pending

    weights_to_load = [
        (MOLMO2_EMBEDDING_KEY, updated_cache["embedding"]),
        (MOLMO2_NEW_EMBEDDING_KEY, updated_cache["new_embedding"]),
        *next_pending,
    ]
    return weights_to_load, updated_cache, []


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        return super().__new__(cls)

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        is_molmo2_standard_load = (
            not peft_config and not is_fp8_model(self.model_runner.vllm_config) and _is_molmo2_vllm_model(self.model_runner.model)
        )
        if is_molmo2_standard_load:
            self._molmo2_pending_weights: list[tuple[str, torch.Tensor]] = []
            if not hasattr(self, "_molmo2_embedding_cache"):
                self._molmo2_embedding_cache: dict[str, torch.Tensor] = {}

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        # build communication buffer
        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(self._get_zmq_handle())

        comm_metadata = socket.recv_pyobj()
        buffer, shm = None, None
        if not use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        socket.send(b"")

        # receive bucket and update weights
        while True:
            metadata = socket.recv_pyobj()
            weights = []
            for name, meta in metadata["bucket_meta"].items():
                shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                size = dtype.itemsize * shape.numel()
                # NOTE: we need to clone the tensor to release CUDA IPC memory
                # but for shared memory, it's not necessary and if we do clone,
                # it will cause extra memory copy overhead and slow down the process.
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                if not use_shm:
                    tensor = tensor.clone()
                else:
                    tensor = tensor.to(self.device)
                weights.append((name, tensor))
            get_torch_device().synchronize()
            socket.send(b"")
            self._update_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
            del weights
            if metadata["is_last"]:
                break

        # clean up
        socket.close()
        del buffer
        if shm is not None:
            shm.close()
            del shm
        if is_molmo2_standard_load and getattr(self, "_molmo2_pending_weights", None):
            pending_count = len(self._molmo2_pending_weights)
            self._molmo2_pending_weights = []
            raise ValueError(
                "Molmo2 weight update finished before the embedding tensors needed for bucketed loading were received. "
                f"Deferred bucket size: {pending_count} tensors."
            )
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                if _is_molmo2_vllm_model(self.model_runner.model):
                    embedding_cache = getattr(self, "_molmo2_embedding_cache", {})
                    pending_weights = getattr(self, "_molmo2_pending_weights", [])
                    weights, embedding_cache, pending_weights = _prepare_molmo2_bucket_weights(
                        self.model_runner.model,
                        weights,
                        embedding_cache=embedding_cache,
                        pending_weights=pending_weights,
                    )
                    self._molmo2_embedding_cache = embedding_cache
                    self._molmo2_pending_weights = pending_weights
                    if weights is None:
                        logger.info("Deferring Molmo2 weight bucket until embedding tensors are available.")
                        return
                logger.info("Loading standard weights (non-FP8, async)")
                self.model_runner.model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        socket_dir = os.getenv("VERL_ZMQ_SOCKET_DIR", tempfile.gettempdir())
        return f"ipc://{socket_dir}/rl-colocate-zmq-{self.device_uuid}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args

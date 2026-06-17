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
import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config, tokenizer=None) -> int:
    def _normalize_len(value):
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        if value > 10_000_000:
            return None
        return value

    def _pick(obj, names):
        if obj is None:
            return None
        for name in names:
            if hasattr(obj, name):
                value = _normalize_len(getattr(obj, name))
                if value is not None:
                    return value
        return None

    def _pick_dict(obj, names):
        for name in names:
            if name in obj:
                value = _normalize_len(obj[name])
                if value is not None:
                    return value
        return None

    primary_candidates = [
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "max_model_len",
        "model_max_length",
        "n_positions",
        "n_ctx",
    ]
    fallback_candidates = ["max_length"]

    max_len = _pick(hf_config, primary_candidates)
    if max_len is None:
        for attr in ("text_config", "language_config", "llm_config"):
            max_len = _pick(getattr(hf_config, attr, None), primary_candidates)
            if max_len is not None:
                break

    if max_len is None and isinstance(hf_config, dict):
        max_len = _pick_dict(hf_config, primary_candidates)

    if max_len is None:
        max_len = _pick(hf_config, fallback_candidates)
        if max_len is None:
            for attr in ("text_config", "language_config", "llm_config"):
                max_len = _pick(getattr(hf_config, attr, None), fallback_candidates)
                if max_len is not None:
                    break
        if max_len is None and isinstance(hf_config, dict):
            max_len = _pick_dict(hf_config, fallback_candidates)

    if max_len is None and tokenizer is not None:
        max_len = _normalize_len(getattr(tokenizer, "model_max_length", None))

    if max_len is None:
        raise ValueError(
            "max_position_embeddings not found in HF config. "
            "Set actor_rollout_ref.rollout.max_model_len explicitly."
        )
    return int(max_len)


async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item

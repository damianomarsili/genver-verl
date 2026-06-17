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

from types import SimpleNamespace

import torch

from verl.workers.rollout.vllm_rollout.utils import (
    MOLMO2_EMBEDDING_KEY,
    MOLMO2_NEW_EMBEDDING_KEY,
    _prepare_molmo2_bucket_weights,
    _split_molmo2_merged_embedding,
)


def _make_molmo2_model(vocab_size: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            model_type="molmo2",
            text_config=SimpleNamespace(model_type="molmo2_text", vocab_size=vocab_size),
        )
    )


def test_split_molmo2_merged_embedding_from_language_model_key() -> None:
    model = _make_molmo2_model(vocab_size=4)
    merged_embedding = torch.arange(12, dtype=torch.float32).view(6, 2)

    normalized = _split_molmo2_merged_embedding(
        [("model.language_model.embed_tokens.weight", merged_embedding)],
        model,
    )

    assert [name for name, _ in normalized] == [MOLMO2_EMBEDDING_KEY, MOLMO2_NEW_EMBEDDING_KEY]
    assert torch.equal(normalized[0][1], merged_embedding[:4])
    assert torch.equal(normalized[1][1], merged_embedding[4:])


def test_prepare_molmo2_bucket_weights_defers_until_embeddings_arrive() -> None:
    model = _make_molmo2_model(vocab_size=4)
    layer_weight = torch.ones(2, 2)

    first_bucket, cache, pending = _prepare_molmo2_bucket_weights(
        model,
        [("model.layers.0.weight", layer_weight)],
        embedding_cache={},
        pending_weights=[],
    )

    assert first_bucket is None
    assert pending == [("model.layers.0.weight", layer_weight)]
    assert cache == {}

    second_bucket, cache, pending = _prepare_molmo2_bucket_weights(
        model,
        [
            (MOLMO2_EMBEDDING_KEY, torch.arange(8, dtype=torch.float32).view(4, 2)),
            (MOLMO2_NEW_EMBEDDING_KEY, torch.arange(4, dtype=torch.float32).view(2, 2)),
        ],
        embedding_cache=cache,
        pending_weights=pending,
    )

    assert second_bucket is not None
    assert [name for name, _ in second_bucket] == [
        MOLMO2_EMBEDDING_KEY,
        MOLMO2_NEW_EMBEDDING_KEY,
        "model.layers.0.weight",
    ]
    assert pending == []
    assert set(cache) == {"embedding", "new_embedding"}


def test_prepare_molmo2_bucket_weights_reuses_cached_embeddings_for_later_buckets() -> None:
    model = _make_molmo2_model(vocab_size=4)
    embedding = torch.arange(8, dtype=torch.float32).view(4, 2)
    new_embedding = torch.arange(4, dtype=torch.float32).view(2, 2)

    first_bucket, cache, pending = _prepare_molmo2_bucket_weights(
        model,
        [
            (MOLMO2_EMBEDDING_KEY, embedding),
            (MOLMO2_NEW_EMBEDDING_KEY, new_embedding),
            ("model.layers.0.weight", torch.ones(2, 2)),
        ],
        embedding_cache={},
        pending_weights=[],
    )

    assert first_bucket is not None
    assert pending == []

    later_bucket, cache, pending = _prepare_molmo2_bucket_weights(
        model,
        [("model.layers.1.weight", torch.zeros(2, 2))],
        embedding_cache=cache,
        pending_weights=pending,
    )

    assert later_bucket is not None
    assert [name for name, _ in later_bucket] == [
        MOLMO2_EMBEDDING_KEY,
        MOLMO2_NEW_EMBEDDING_KEY,
        "model.layers.1.weight",
    ]
    assert torch.equal(later_bucket[0][1], embedding)
    assert torch.equal(later_bucket[1][1], new_embedding)
    assert pending == []

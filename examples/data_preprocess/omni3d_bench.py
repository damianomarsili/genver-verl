#!/usr/bin/env python3
"""
Preprocess Omni3D-Bench into VERL-compatible parquet files.

This mirrors the gsm8k preprocessor structure and stores answer_type for reward computation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Omni3DExample:
    prompt: list[dict]
    answer: str
    answer_type: str
    images: list[dict]
    data_source: str
    extra_info: dict


def _build_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"<image>\n{question.strip()}",
        }
    ]


def _to_image_entry(image: dict) -> dict:
    if "bytes" in image:
        return {"bytes": image["bytes"]}
    if "path" in image:
        return {"path": image["path"]}
    return image


def _load_split(dataset_name: str, split: str, max_samples: int | None = None):
    dataset = datasets.load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _convert_dataset(dataset, data_source: str) -> list[Omni3DExample]:
    outputs: list[Omni3DExample] = []
    for row in dataset:
        question = row.get("question") or row.get("query") or row.get("prompt")
        if question is None:
            raise ValueError("Missing question field in dataset row.")

        answer = row.get("answer") or row.get("label") or row.get("target")
        if answer is None:
            answer = ""

        answer_type = row.get("answer_type") or row.get("answerType") or row.get("type")
        if answer_type is None:
            answer_type = "string"

        images = row.get("images") or []
        if isinstance(images, dict):
            images = [images]
        images = [_to_image_entry(image) for image in images]

        extra_info = {k: row[k] for k in row.keys() if k not in {"question", "query", "prompt", "answer", "label", "target", "answer_type", "answerType", "type", "images"}}
        outputs.append(
            Omni3DExample(
                prompt=_build_prompt(question),
                answer=str(answer),
                answer_type=str(answer_type),
                images=images,
                data_source=data_source,
                extra_info=extra_info,
            )
        )
    return outputs


def _write_parquet(rows: list[Omni3DExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "prompt": row.prompt,
                "answer": row.answer,
                "answer_type": row.answer_type,
                "images": row.images,
                "data_source": row.data_source,
                "extra_info": row.extra_info,
            }
            for row in rows
        ]
    )
    pq.write_table(table, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Omni3D-Bench for VERL.")
    parser.add_argument("--dataset", default="dmarsili/Omni3D-Bench")
    parser.add_argument("--output_dir", default="data/omni3d-bench")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="train")
    parser.add_argument("--test_size", type=int, default=50)
    args = parser.parse_args()

    train_dataset = _load_split(args.dataset, args.train_split)
    test_dataset = _load_split(args.dataset, args.test_split, max_samples=args.test_size)

    train_rows = _convert_dataset(train_dataset, data_source="omni3d_bench")
    test_rows = _convert_dataset(test_dataset, data_source="omni3d_bench")

    output_dir = Path(args.output_dir)
    _write_parquet(train_rows, output_dir / "train.parquet")
    _write_parquet(test_rows, output_dir / "test.parquet")


if __name__ == "__main__":
    main()

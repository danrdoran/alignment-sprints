"""Dataset loading utilities for TruthfulQA mini ICM experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class TruthfulQAExample:
    """Single claim example in TruthfulQA mini format."""

    uid: int
    question: str
    choice: str
    label: int
    consistency_id: int
    split: str

    @property
    def label_text(self) -> str:
        return "True" if self.label == 1 else "False"


def _validate_row(row: dict, index: int, split: str) -> None:
    required = {"question", "choice", "label", "consistency_id"}
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"{split}[{index}] missing keys: {missing}")

    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{split}[{index}] has invalid question")
    if not isinstance(row["choice"], str) or not row["choice"].strip():
        raise ValueError(f"{split}[{index}] has invalid choice")
    if row["label"] not in (0, 1):
        raise ValueError(f"{split}[{index}] label must be 0/1, got {row['label']!r}")
    if not isinstance(row["consistency_id"], int):
        raise ValueError(
            f"{split}[{index}] consistency_id must be int, got {type(row['consistency_id']).__name__}"
        )


def load_truthfulqa_split(path: str | Path, split: str) -> List[TruthfulQAExample]:
    """Load a single split file from JSON list format."""
    split_name = split.strip().lower()
    if split_name not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    data = json.loads(file_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {file_path}, got {type(data).__name__}")

    examples: List[TruthfulQAExample] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{split_name}[{idx}] must be object, got {type(row).__name__}")
        _validate_row(row, idx, split_name)
        examples.append(
            TruthfulQAExample(
                uid=idx,
                question=row["question"],
                choice=row["choice"],
                label=int(row["label"]),
                consistency_id=int(row["consistency_id"]),
                split=split_name,
            )
        )
    return examples


def load_truthfulqa_dataset(
    train_path: str | Path = "data/truthfulqa-mini/truthfulqa_train.json",
    test_path: str | Path = "data/truthfulqa-mini/truthfulqa_test.json",
) -> Dict[str, List[TruthfulQAExample]]:
    """Load train and test splits for TruthfulQA mini."""
    train_examples = load_truthfulqa_split(train_path, split="train")
    test_examples = load_truthfulqa_split(test_path, split="test")
    return {"train": train_examples, "test": test_examples}


def group_by_consistency_id(
    examples: Iterable[TruthfulQAExample],
) -> Dict[int, List[TruthfulQAExample]]:
    """Group examples by consistency_id."""
    grouped: Dict[int, List[TruthfulQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.consistency_id, []).append(ex)
    return grouped

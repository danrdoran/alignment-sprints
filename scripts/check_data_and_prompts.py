#!/usr/bin/env python3
"""Sanity checks for Step 4: dataset loader + prompt builders."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.data import group_by_consistency_id, load_truthfulqa_dataset
from icm_ue.prompts import build_example_prompt, build_fewshot_prompt_from_examples


def summarize_split(name: str, examples: list) -> dict:
    groups = group_by_consistency_id(examples)
    group_size_dist = Counter(len(v) for v in groups.values())
    label_dist = Counter(ex.label for ex in examples)
    return {
        "split": name,
        "num_examples": len(examples),
        "num_groups": len(groups),
        "group_size_distribution": dict(sorted(group_size_dist.items())),
        "label_distribution": dict(sorted(label_dist.items())),
    }


def main() -> int:
    dataset = load_truthfulqa_dataset()
    train_examples = dataset["train"]
    test_examples = dataset["test"]

    train_summary = summarize_split("train", train_examples)
    test_summary = summarize_split("test", test_examples)

    demo_prompt = build_example_prompt(train_examples[0])
    fewshot_prompt = build_fewshot_prompt_from_examples(
        target=train_examples[2],
        demonstration_examples=[train_examples[0], train_examples[1]],
    )

    print("Step 4 sanity check")
    print(f"Train summary: {train_summary}")
    print(f"Test summary: {test_summary}")
    print("\nSingle example prompt preview:")
    print(demo_prompt)
    print("\nFew-shot prompt preview (2 demos + 1 target):")
    print(fewshot_prompt)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path("outputs/raw") / f"step4_data_prompt_sanity_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": timestamp,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "single_prompt_preview": demo_prompt,
        "fewshot_prompt_preview": fewshot_prompt,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

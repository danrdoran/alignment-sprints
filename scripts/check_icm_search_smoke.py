#!/usr/bin/env python3
"""Offline smoke test for Step 7 ICM search logic (no API calls)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.data import load_truthfulqa_dataset
from icm_ue.icm_search import ICMConfig, run_icm_search


@dataclass
class _FakeScore:
    score: float
    predicted_label_int: int


class FakeQuerier:
    """Deterministic fake querier for fast local validation."""

    def score_target(self, target, demonstrations):
        # Cheap deterministic pseudo-score: sum demo labels pushes prediction.
        balance = sum(label for _, label in demonstrations) - (len(demonstrations) / 2)
        score = balance * 0.2
        return _FakeScore(score=score, predicted_label_int=1 if score > 0 else 0)


def main() -> int:
    dataset = load_truthfulqa_dataset()
    train = dataset["train"]
    cfg = ICMConfig(max_examples=24, max_iterations=10, num_seed=6, seed=7)
    result = run_icm_search(train, querier=FakeQuerier(), cfg=cfg)

    labeled = sum(v is not None for v in result.final_assignments.values())
    print("Step 7 smoke test")
    print(f"history_len={len(result.history)} labeled={labeled}")
    print(f"current_utility={result.current_utility:.6f} best_utility={result.best_utility:.6f}")
    if result.history:
        print(
            f"first_iter_target={result.history[0].target_uid} "
            f"last_iter_target={result.history[-1].target_uid}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

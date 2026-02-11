#!/usr/bin/env python3
"""Run ICM search across multiple seeds and save pseudo-label artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.data import load_truthfulqa_dataset
from icm_ue.icm_search import ICMConfig, run_icm_search
from icm_ue.querying import ICMBaseModelQuerier


@dataclass(frozen=True)
class SeedSummary:
    seed: int
    selected_examples: int
    labeled_count: int
    unlabeled_count: int
    labeled_fraction: float
    current_utility: float
    best_utility: float
    label_distribution: dict
    agreement_with_gold: float | None
    run_log_path: str
    labels_path: str


class _FakeScore:
    def __init__(self, score: float, predicted_label_int: int):
        self.score = score
        self.predicted_label_int = predicted_label_int


class FakeQuerier:
    """Deterministic fake querier for smoke checks and fast script validation."""

    def score_target(self, target, demonstrations):
        balance = sum(label for _, label in demonstrations) - (len(demonstrations) / 2)
        score = float(balance * 0.2)
        return _FakeScore(score=score, predicted_label_int=1 if score > 0 else 0)


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _label_to_text(label: int | None) -> str | None:
    if label is None:
        return None
    return "True" if int(label) == 1 else "False"


def _build_label_rows(train_by_uid, selected_uids: Sequence[int], assignments: dict) -> list[dict]:
    rows: list[dict] = []
    for uid in selected_uids:
        ex = train_by_uid[uid]
        label = assignments.get(uid)
        rows.append(
            {
                "uid": ex.uid,
                "question": ex.question,
                "choice": ex.choice,
                "consistency_id": ex.consistency_id,
                "gold_label": ex.label,
                "gold_label_text": _label_to_text(ex.label),
                "icm_label": label,
                "icm_label_text": _label_to_text(label),
                "is_labeled": label is not None,
            }
        )
    return rows


def _summarize_rows(rows: Iterable[dict]) -> tuple[int, int, dict, float | None]:
    rows = list(rows)
    labeled = [row for row in rows if row["is_labeled"]]
    unlabeled_count = len(rows) - len(labeled)
    label_dist = Counter(row["icm_label"] for row in labeled)

    if not labeled:
        agreement = None
    else:
        agreement = sum(int(row["icm_label"] == row["gold_label"]) for row in labeled) / len(labeled)

    return len(labeled), unlabeled_count, dict(sorted(label_dist.items())), agreement


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ICM search over multiple seeds and save label artifacts."
    )
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--num-seed", type=int, default=8)
    parser.add_argument("--max-demos", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--initial-temperature", type=float, default=10.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)
    parser.add_argument("--cooling-beta", type=float, default=0.99)
    parser.add_argument("--sample-priority-weight", type=float, default=100.0)
    parser.add_argument("--labels-from", choices=["final", "best"], default="best")
    parser.add_argument("--dry-run-fake", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    dataset = load_truthfulqa_dataset()
    train = dataset["train"]
    train_by_uid = {ex.uid: ex for ex in train}

    querier = FakeQuerier() if args.dry_run_fake else ICMBaseModelQuerier.from_env()
    model_name = "fake-querier" if args.dry_run_fake else querier.settings.model

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_seed_summaries: list[SeedSummary] = []

    for seed in seeds:
        cfg = ICMConfig(
            alpha=args.alpha,
            num_seed=args.num_seed,
            max_iterations=args.max_iterations,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            cooling_beta=args.cooling_beta,
            max_examples=args.max_examples,
            max_demos=args.max_demos,
            sample_priority_weight=args.sample_priority_weight,
            seed=seed,
        )
        run_result = run_icm_search(examples=train, querier=querier, cfg=cfg)

        chosen_assignments = (
            run_result.best_assignments if args.labels_from == "best" else run_result.final_assignments
        )
        rows = _build_label_rows(train_by_uid, run_result.selected_uids, chosen_assignments)
        labeled_count, unlabeled_count, label_dist, agreement = _summarize_rows(rows)

        run_log_path = Path("outputs/logs") / f"icm_search_seed{seed}_{timestamp}.json"
        labels_path = Path("outputs/labels") / f"icm_labels_seed{seed}_{timestamp}.json"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)

        run_log_path.write_text(json.dumps(run_result.to_jsonable(), indent=2))
        labels_path.write_text(json.dumps(rows, indent=2))

        labeled_fraction = 0.0 if len(rows) == 0 else labeled_count / len(rows)
        summary = SeedSummary(
            seed=seed,
            selected_examples=len(rows),
            labeled_count=labeled_count,
            unlabeled_count=unlabeled_count,
            labeled_fraction=labeled_fraction,
            current_utility=run_result.current_utility,
            best_utility=run_result.best_utility,
            label_distribution=label_dist,
            agreement_with_gold=agreement,
            run_log_path=str(run_log_path),
            labels_path=str(labels_path),
        )
        per_seed_summaries.append(summary)

        print(
            f"seed={seed} selected={summary.selected_examples} labeled={summary.labeled_count} "
            f"unlabeled={summary.unlabeled_count} best_utility={summary.best_utility:.4f} "
            f"agreement={summary.agreement_with_gold if summary.agreement_with_gold is not None else 'n/a'}"
        )

    summary_path = Path("outputs/metrics") / f"icm_multiseed_summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "timestamp_utc": timestamp,
        "model": model_name,
        "labels_from": args.labels_from,
        "config": {
            "max_examples": args.max_examples,
            "max_iterations": args.max_iterations,
            "num_seed": args.num_seed,
            "max_demos": args.max_demos,
            "alpha": args.alpha,
            "initial_temperature": args.initial_temperature,
            "final_temperature": args.final_temperature,
            "cooling_beta": args.cooling_beta,
            "sample_priority_weight": args.sample_priority_weight,
            "seeds": seeds,
            "dry_run_fake": args.dry_run_fake,
        },
        "per_seed": [asdict(item) for item in per_seed_summaries],
    }
    summary_path.write_text(json.dumps(aggregate, indent=2))

    print(f"saved_summary={summary_path}")
    print(f"saved_summary_abs={summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

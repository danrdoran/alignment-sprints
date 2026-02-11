#!/usr/bin/env python3
"""Run Step 7 ICM search loop (without logical consistency fix)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.data import load_truthfulqa_dataset
from icm_ue.icm_search import ICMConfig, run_icm_search
from icm_ue.querying import ICMBaseModelQuerier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ICM search loop on TruthfulQA train split (no consistency fix)."
    )
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--num-seed", type=int, default=8)
    parser.add_argument("--max-demos", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--initial-temperature", type=float, default=10.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)
    parser.add_argument("--cooling-beta", type=float, default=0.99)
    parser.add_argument("--sample-priority-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dataset = load_truthfulqa_dataset()
    train = dataset["train"]
    querier = ICMBaseModelQuerier.from_env()

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
        seed=args.seed,
    )

    result = run_icm_search(examples=train, querier=querier, cfg=cfg)

    labeled_count = sum(v is not None for v in result.final_assignments.values())
    print("Step 7 ICM search run")
    print(f"selected_examples={len(result.selected_uids)}")
    print(f"iterations={len(result.history)}")
    print(f"final_labeled_count={labeled_count}")
    print(f"current_utility={result.current_utility:.6f}")
    print(f"best_utility={result.best_utility:.6f}")
    if result.history:
        last = result.history[-1]
        print(
            "last_iter="
            f"{last.iteration} target_uid={last.target_uid} "
            f"proposed={last.proposed_label} accepted={last.accepted} "
            f"delta={last.delta:.6f}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output) if args.output else Path("outputs/logs") / f"icm_search_run_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_jsonable(), indent=2))
    print(f"saved_artifact={out_path}")
    print(f"saved_artifact_abs={out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sanity check for Step 6 base-model query helper."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.data import load_truthfulqa_dataset
from icm_ue.querying import ICMBaseModelQuerier


def pick_balanced_demos(train):
    """
    Pick a target and two demonstrations with one True and one False label.

    Preference:
    - choose both demos from the target's consistency group (if available)
    - otherwise fallback to first True/False in train split
    """
    target = train[2]
    same_group = [ex for ex in train if ex.consistency_id == target.consistency_id and ex.uid != target.uid]
    true_demo = next((ex for ex in same_group if ex.label == 1), None)
    false_demo = next((ex for ex in same_group if ex.label == 0), None)

    if true_demo is None:
        true_demo = next(ex for ex in train if ex.label == 1 and ex.uid != target.uid)
    if false_demo is None:
        false_demo = next(ex for ex in train if ex.label == 0 and ex.uid != target.uid and ex.uid != true_demo.uid)

    return target, [(true_demo, true_demo.label), (false_demo, false_demo.label)]


def main() -> int:
    dataset = load_truthfulqa_dataset()
    train = dataset["train"]

    target, demonstrations = pick_balanced_demos(train)

    querier = ICMBaseModelQuerier.from_env()
    result = querier.score_target(target=target, demonstrations=demonstrations)

    print("Step 6 query helper sanity")
    print(f"model={result.model}")
    print(f"target_uid={result.target_uid}")
    print(f"completion={result.completion!r}")
    print(f"score={result.score:.6f}")
    print(f"predicted_label={result.predicted_label_text} ({result.predicted_label_int})")
    print(f"prob_true={result.prob_true:.6f} prob_false={result.prob_false:.6f}")
    print(f"signal_mass={result.signal_mass:.6f} valid_signal={result.valid_signal}")
    print(f"attempt={result.attempt}")
    print(f"demo_labels={[label for _, label in demonstrations]}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path("outputs/raw") / f"step6_query_helper_sanity_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": timestamp,
        "result": asdict(result),
        "demonstration_uids": [demo.uid for demo, _ in demonstrations],
        "demonstration_labels": [int(label) for _, label in demonstrations],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved_artifact={out_path}")
    print(f"saved_artifact_abs={out_path.resolve()}")
    print(f"saved_exists={out_path.exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

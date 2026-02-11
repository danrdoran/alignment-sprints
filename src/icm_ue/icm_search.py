"""ICM search loop implementation (without logical consistency fix)."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .data import TruthfulQAExample


@dataclass(frozen=True)
class ICMConfig:
    """Configuration for the simulated-annealing ICM search loop."""

    alpha: float = 50.0
    num_seed: int = 8
    max_iterations: int = 30
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_beta: float = 0.99
    max_examples: int = 64
    max_demos: int = 24
    sample_priority_weight: float = 100.0
    seed: int = 42


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration search diagnostics."""

    iteration: int
    target_uid: int
    current_label: Optional[int]
    proposed_label: int
    temperature: float
    current_utility: float
    candidate_utility: float
    delta: float
    accept_prob: float
    accepted: bool
    labeled_count: int
    best_utility: float


@dataclass(frozen=True)
class ICMRunResult:
    """Final output for one ICM run."""

    config: ICMConfig
    selected_uids: List[int]
    current_utility: float
    best_utility: float
    final_assignments: Dict[int, Optional[int]]
    best_assignments: Dict[int, Optional[int]]
    history: List[IterationRecord]

    def to_jsonable(self) -> dict:
        return {
            "config": asdict(self.config),
            "selected_uids": self.selected_uids,
            "current_utility": self.current_utility,
            "best_utility": self.best_utility,
            "final_assignments": self.final_assignments,
            "best_assignments": self.best_assignments,
            "history": [asdict(item) for item in self.history],
        }


def _temperature_at(iteration: int, cfg: ICMConfig) -> float:
    # Matches paper/log-style update: T = max(Tmin, T0 / (1 + beta*log(n)))
    return max(
        cfg.final_temperature,
        cfg.initial_temperature / (1 + cfg.cooling_beta * math.log(max(1, iteration))),
    )


def _select_examples(
    examples: Sequence[TruthfulQAExample], max_examples: int, seed: int
) -> List[TruthfulQAExample]:
    if max_examples <= 0 or max_examples >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    selected = rng.sample(list(examples), max_examples)
    return sorted(selected, key=lambda ex: ex.uid)


def _initialize_assignments(
    examples: Sequence[TruthfulQAExample], cfg: ICMConfig
) -> Dict[int, Optional[int]]:
    rng = random.Random(cfg.seed)
    uids = [ex.uid for ex in examples]
    if len(uids) == 0:
        return {}

    k = min(cfg.num_seed, len(uids))
    seed_uids = set(rng.sample(uids, k))
    assignments: Dict[int, Optional[int]] = {uid: None for uid in uids}

    # Balanced random seed labels.
    seed_labels = [1] * (k // 2) + [0] * (k // 2)
    if len(seed_labels) < k:
        seed_labels.append(rng.choice([0, 1]))
    rng.shuffle(seed_labels)

    for uid, label in zip(sorted(seed_uids), seed_labels):
        assignments[uid] = int(label)
    return assignments


def _build_group_index(examples: Sequence[TruthfulQAExample]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for ex in examples:
        groups.setdefault(ex.consistency_id, []).append(ex.uid)
    for gid in groups:
        groups[gid] = sorted(groups[gid])
    return groups


def _build_demos_for_target(
    target_uid: int,
    assignments: Mapping[int, Optional[int]],
    examples_by_uid: Mapping[int, TruthfulQAExample],
    groups_by_consistency: Mapping[int, List[int]],
    max_demos: int,
) -> List[Tuple[TruthfulQAExample, int]]:
    target_ex = examples_by_uid[target_uid]
    group_uids = set(groups_by_consistency.get(target_ex.consistency_id, []))

    same_group: List[Tuple[TruthfulQAExample, int]] = []
    other: List[Tuple[TruthfulQAExample, int]] = []
    for uid, label in sorted(assignments.items()):
        if uid == target_uid or label is None:
            continue
        pair = (examples_by_uid[uid], int(label))
        if uid in group_uids:
            same_group.append(pair)
        else:
            other.append(pair)

    demos = same_group + other
    if max_demos > 0:
        demos = demos[:max_demos]
    return demos


def _sample_target_uid(
    assignments: Mapping[int, Optional[int]],
    examples_by_uid: Mapping[int, TruthfulQAExample],
    groups_by_consistency: Mapping[int, List[int]],
    rng: random.Random,
    priority_weight: float,
) -> int:
    uids = sorted(assignments.keys())
    weights: List[float] = []
    for uid in uids:
        label = assignments[uid]
        if label is not None:
            weights.append(1.0)
            continue

        # Prioritize unlabeled examples with labeled peers in same consistency group.
        gid = examples_by_uid[uid].consistency_id
        has_labeled_peer = any(
            (other_uid != uid) and (assignments.get(other_uid) is not None)
            for other_uid in groups_by_consistency.get(gid, [])
        )
        weights.append(priority_weight if has_labeled_peer else 1.0)

    return rng.choices(uids, weights=weights, k=1)[0]


def _compute_mutual_predictability_utility(
    assignments: Mapping[int, Optional[int]],
    querier,
    examples_by_uid: Mapping[int, TruthfulQAExample],
    groups_by_consistency: Mapping[int, List[int]],
    cfg: ICMConfig,
    contribution_cache: Dict[Tuple[int, int, Tuple[Tuple[int, int], ...]], float],
) -> float:
    labeled_uids = [uid for uid, label in assignments.items() if label is not None]
    if not labeled_uids:
        return -1e9

    contributions: List[float] = []
    for uid in sorted(labeled_uids):
        label = int(assignments[uid])
        demos = _build_demos_for_target(
            target_uid=uid,
            assignments=assignments,
            examples_by_uid=examples_by_uid,
            groups_by_consistency=groups_by_consistency,
            max_demos=cfg.max_demos,
        )
        if not demos:
            continue

        demo_signature = tuple((demo_ex.uid, int(demo_label)) for demo_ex, demo_label in demos)
        cache_key = (uid, label, demo_signature)
        if cache_key in contribution_cache:
            contributions.append(contribution_cache[cache_key])
            continue

        result = querier.score_target(target=examples_by_uid[uid], demonstrations=demos)
        contribution = result.score if label == 1 else -result.score
        contribution_cache[cache_key] = contribution
        contributions.append(contribution)

    if not contributions:
        return -1e9
    # Without consistency fix/penalty term: utility is scaled mutual predictability.
    return cfg.alpha * (sum(contributions) / len(contributions))


def run_icm_search(
    examples: Sequence[TruthfulQAExample],
    querier,
    cfg: ICMConfig,
) -> ICMRunResult:
    """
    Run Algorithm-1-style search with simulated annealing, without consistency-fix.

    Notes:
    - Does not apply logical consistency fix (per assignment requirement).
    - Uses only mutual predictability term for utility.
    """
    selected_examples = _select_examples(examples, max_examples=cfg.max_examples, seed=cfg.seed)
    examples_by_uid = {ex.uid: ex for ex in selected_examples}
    groups_by_consistency = _build_group_index(selected_examples)
    rng = random.Random(cfg.seed)

    assignments = _initialize_assignments(selected_examples, cfg)
    best_assignments = dict(assignments)
    contribution_cache: Dict[Tuple[int, int, Tuple[Tuple[int, int], ...]], float] = {}

    current_utility = _compute_mutual_predictability_utility(
        assignments=assignments,
        querier=querier,
        examples_by_uid=examples_by_uid,
        groups_by_consistency=groups_by_consistency,
        cfg=cfg,
        contribution_cache=contribution_cache,
    )
    best_utility = current_utility

    history: List[IterationRecord] = []
    for iteration in range(1, cfg.max_iterations + 1):
        temperature = _temperature_at(iteration, cfg)
        target_uid = _sample_target_uid(
            assignments=assignments,
            examples_by_uid=examples_by_uid,
            groups_by_consistency=groups_by_consistency,
            rng=rng,
            priority_weight=cfg.sample_priority_weight,
        )
        current_label = assignments[target_uid]

        demos = _build_demos_for_target(
            target_uid=target_uid,
            assignments=assignments,
            examples_by_uid=examples_by_uid,
            groups_by_consistency=groups_by_consistency,
            max_demos=cfg.max_demos,
        )

        if demos:
            proposed_label = int(
                querier.score_target(
                    target=examples_by_uid[target_uid],
                    demonstrations=demos,
                ).predicted_label_int
            )
        else:
            # No demonstrations yet: random bootstrap proposal.
            proposed_label = rng.choice([0, 1])

        candidate_assignments = dict(assignments)
        candidate_assignments[target_uid] = proposed_label

        candidate_utility = _compute_mutual_predictability_utility(
            assignments=candidate_assignments,
            querier=querier,
            examples_by_uid=examples_by_uid,
            groups_by_consistency=groups_by_consistency,
            cfg=cfg,
            contribution_cache=contribution_cache,
        )

        delta = candidate_utility - current_utility
        if delta > 0:
            accept_prob = 1.0
            accepted = True
        else:
            accept_prob = math.exp(delta / max(temperature, 1e-9))
            accepted = rng.random() < accept_prob

        if accepted:
            assignments = candidate_assignments
            current_utility = candidate_utility

        if current_utility > best_utility:
            best_utility = current_utility
            best_assignments = dict(assignments)

        labeled_count = sum(label is not None for label in assignments.values())
        history.append(
            IterationRecord(
                iteration=iteration,
                target_uid=target_uid,
                current_label=current_label,
                proposed_label=proposed_label,
                temperature=temperature,
                current_utility=current_utility,
                candidate_utility=candidate_utility,
                delta=delta,
                accept_prob=accept_prob,
                accepted=accepted,
                labeled_count=labeled_count,
                best_utility=best_utility,
            )
        )

    return ICMRunResult(
        config=cfg,
        selected_uids=[ex.uid for ex in selected_examples],
        current_utility=current_utility,
        best_utility=best_utility,
        final_assignments=assignments,
        best_assignments=best_assignments,
        history=history,
    )


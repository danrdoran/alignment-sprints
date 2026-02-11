"""Core package for TruthfulQA mini ICM replication."""

from .data import (
    TruthfulQAExample,
    group_by_consistency_id,
    load_truthfulqa_dataset,
    load_truthfulqa_split,
)
from .prompts import (
    TRUTHFULQA_TEMPLATE,
    build_claim_prompt,
    build_example_prompt,
    build_fewshot_prompt,
    build_fewshot_prompt_from_examples,
    build_labeled_example_line,
    label_to_text,
)
from .querying import ICMBaseModelQuerier, QueryScoreResult, QuerySettings
from .icm_search import ICMConfig, ICMRunResult, IterationRecord, run_icm_search
from .scoring import (
    DEFAULT_EPS,
    DEFAULT_MIN_SIGNAL_MASS,
    TrueFalseScore,
    normalize_top_logprobs,
    score_true_false_from_top_logprobs,
    token_to_true_false,
)

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_MIN_SIGNAL_MASS",
    "TRUTHFULQA_TEMPLATE",
    "TruthfulQAExample",
    "TrueFalseScore",
    "build_claim_prompt",
    "build_example_prompt",
    "build_fewshot_prompt",
    "build_fewshot_prompt_from_examples",
    "build_labeled_example_line",
    "group_by_consistency_id",
    "ICMConfig",
    "ICMBaseModelQuerier",
    "ICMRunResult",
    "IterationRecord",
    "label_to_text",
    "load_truthfulqa_dataset",
    "load_truthfulqa_split",
    "normalize_top_logprobs",
    "QueryScoreResult",
    "QuerySettings",
    "run_icm_search",
    "score_true_false_from_top_logprobs",
    "token_to_true_false",
]

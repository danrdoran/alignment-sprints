"""Scoring utilities for ICM label inference from token logprobs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_EPS = 1e-5
DEFAULT_MIN_SIGNAL_MASS = 1e-4


@dataclass(frozen=True)
class TrueFalseScore:
    """Score summary for binary True/False inference."""

    score: float
    prob_true: float
    prob_false: float
    signal_mass: float
    valid_signal: bool

    @property
    def predicted_label_text(self) -> str:
        return "True" if self.score > 0 else "False"

    @property
    def predicted_label_int(self) -> int:
        return 1 if self.score > 0 else 0


def token_to_true_false(token: str) -> bool | None:
    """
    Map a token string to True/False.

    Returns:
    - True for tokens that indicate true
    - False for tokens that indicate false
    - None for tokens not clearly indicating either
    """
    token_lower = token.lower()
    has_true = "true" in token_lower
    has_false = "false" in token_lower
    if has_true == has_false:
        return None
    return has_true


def normalize_top_logprobs(choice: Any) -> dict[str, float]:
    """
    Normalize completion choice logprobs into `token -> logprob`.

    Handles common response shapes across OpenAI-compatible providers.
    """
    out: dict[str, float] = {}

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return out

    top_logprobs = getattr(logprobs, "top_logprobs", None)
    if not top_logprobs:
        return out

    first = top_logprobs[0]

    # Shape 1: dict[token] = logprob
    if isinstance(first, dict):
        for token, logprob in first.items():
            try:
                out[str(token)] = float(logprob)
            except Exception:  # noqa: BLE001
                continue
        return out

    # Shape 2: list of objects with token/logprob
    if isinstance(first, list):
        for item in first:
            token = getattr(item, "token", None)
            logprob = getattr(item, "logprob", None)
            if token is None or logprob is None:
                continue
            try:
                out[str(token)] = float(logprob)
            except Exception:  # noqa: BLE001
                continue
        return out

    # Shape 3: object-like record
    token = getattr(first, "token", None)
    logprob = getattr(first, "logprob", None)
    if token is not None and logprob is not None:
        try:
            out[str(token)] = float(logprob)
        except Exception:  # noqa: BLE001
            pass

    return out


def score_true_false_from_top_logprobs(
    top_logprobs: Mapping[str, float],
    *,
    eps: float = DEFAULT_EPS,
    min_signal_mass: float = DEFAULT_MIN_SIGNAL_MASS,
) -> TrueFalseScore:
    """
    Compute ICM binary score: logP(True) - logP(False).

    Args:
    - top_logprobs: token -> logprob map for the first generated token
    - eps: additive smoothing for empty-mass stability
    - min_signal_mass: required probability mass on True/False tokens
    """
    prob_sums = {True: eps, False: eps}
    for token, logprob in top_logprobs.items():
        label = token_to_true_false(token)
        if label is None:
            continue
        prob_sums[label] += math.exp(float(logprob))

    prob_true = prob_sums[True]
    prob_false = prob_sums[False]
    score = math.log(prob_true) - math.log(prob_false)
    signal_mass = max(0.0, prob_true + prob_false - (2 * eps))
    valid_signal = signal_mass >= min_signal_mass
    return TrueFalseScore(
        score=score,
        prob_true=prob_true,
        prob_false=prob_false,
        signal_mass=signal_mass,
        valid_signal=valid_signal,
    )


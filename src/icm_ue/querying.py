"""Model querying helpers for ICM label search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI

from .data import TruthfulQAExample
from .prompts import build_fewshot_prompt
from .scoring import (
    DEFAULT_MIN_SIGNAL_MASS,
    normalize_top_logprobs,
    score_true_false_from_top_logprobs,
)


DEFAULT_BASE_URL = "https://api.hyperbolic.xyz/v1"
DEFAULT_SCORING_PREFIX = (
    "You are a binary truthfulness classifier.\n"
    "Output exactly one word: True or False.\n\n"
)


@dataclass(frozen=True)
class QuerySettings:
    """Configuration for base-model label inference calls."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: int = 20
    timeout: float = 120.0
    min_signal_mass: float = DEFAULT_MIN_SIGNAL_MASS
    max_retries: int = 3
    prepend_scoring_prefix: bool = True


@dataclass(frozen=True)
class QueryScoreResult:
    """Structured result for one target-example scoring query."""

    target_uid: int
    model: str
    prompt: str
    completion: str
    attempt: int
    score: float
    predicted_label_int: int
    predicted_label_text: str
    prob_true: float
    prob_false: float
    signal_mass: float
    valid_signal: bool
    top_logprobs: dict[str, float]


class ICMBaseModelQuerier:
    """Reusable wrapper around OpenAI-compatible completion API for ICM scoring."""

    def __init__(self, client: OpenAI, settings: QuerySettings):
        self.client = client
        self.settings = settings

    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> "ICMBaseModelQuerier":
        """Create querier from `.env` values used across this project."""
        if load_dotenv_file:
            load_dotenv()

        api_key = os.getenv("HYPERBOLIC_API_KEY")
        if not api_key:
            raise ValueError("Missing HYPERBOLIC_API_KEY in environment.")

        model = os.getenv("HYPERBOLIC_BASE_MODEL", "meta-llama/Meta-Llama-3.1-405B")
        base_url = os.getenv("HYPERBOLIC_BASE_URL", DEFAULT_BASE_URL)
        client = OpenAI(api_key=api_key, base_url=base_url)
        settings = QuerySettings(model=model, base_url=base_url)
        return cls(client=client, settings=settings)

    def _build_prompt(
        self,
        target: TruthfulQAExample,
        demonstrations: Sequence[tuple[TruthfulQAExample, int | bool]],
    ) -> str:
        core_prompt = build_fewshot_prompt(target=target, demonstrations=demonstrations)
        if self.settings.prepend_scoring_prefix:
            return DEFAULT_SCORING_PREFIX + core_prompt
        return core_prompt

    def score_target(
        self,
        target: TruthfulQAExample,
        demonstrations: Sequence[tuple[TruthfulQAExample, int | bool]],
    ) -> QueryScoreResult:
        """
        Score a single target example using in-context demonstrations.

        Retries if True/False token mass is too low (`valid_signal=False`).
        """
        if len(demonstrations) == 0:
            raise ValueError("At least one demonstration is required for score_target().")

        prompt = self._build_prompt(target=target, demonstrations=demonstrations)
        last_result: QueryScoreResult | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self.client.completions.create(
                    model=self.settings.model,
                    prompt=prompt,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    logprobs=self.settings.logprobs,
                    timeout=self.settings.timeout,
                )
            except (APIStatusError, APIError, Exception) as exc:
                if attempt >= self.settings.max_retries:
                    raise RuntimeError(
                        f"Completion request failed after {attempt} attempts for target uid={target.uid}: {exc}"
                    ) from exc
                continue

            choice = response.choices[0]
            completion = (choice.text or "").strip()
            top_logprobs = normalize_top_logprobs(choice)
            score_obj = score_true_false_from_top_logprobs(
                top_logprobs,
                min_signal_mass=self.settings.min_signal_mass,
            )

            last_result = QueryScoreResult(
                target_uid=target.uid,
                model=self.settings.model,
                prompt=prompt,
                completion=completion,
                attempt=attempt,
                score=score_obj.score,
                predicted_label_int=score_obj.predicted_label_int,
                predicted_label_text=score_obj.predicted_label_text,
                prob_true=score_obj.prob_true,
                prob_false=score_obj.prob_false,
                signal_mass=score_obj.signal_mass,
                valid_signal=score_obj.valid_signal,
                top_logprobs=top_logprobs,
            )

            if score_obj.valid_signal:
                return last_result

        if last_result is None:
            raise RuntimeError(f"No successful completion responses for target uid={target.uid}.")
        return last_result


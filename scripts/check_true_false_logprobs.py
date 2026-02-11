#!/usr/bin/env python3
"""Sanity-check True/False logprob extraction for ICM scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI

# Allow `python scripts/...` execution without manual PYTHONPATH setup.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icm_ue.scoring import (
    DEFAULT_MIN_SIGNAL_MASS,
    normalize_top_logprobs,
    score_true_false_from_top_logprobs,
)


DEFAULT_BASE_URL = "https://api.hyperbolic.xyz/v1"


@dataclass
class PromptResult:
    name: str
    prompt: str
    completion: str
    score: float
    predicted_label: str
    prob_true: float
    prob_false: float
    signal_mass: float
    valid_signal: bool
    attempt: int
    top_logprobs: dict[str, float]


def build_prompt_variants(question: str, claim: str) -> list[str]:
    shared_instruction = (
        "You are a binary truthfulness classifier.\n"
        "Output exactly one word: True or False.\n\n"
    )

    fewshot_block = (
        "Question: What is 2 + 2?\n"
        "Claim: 2 + 2 = 4.\n"
        "I think this Claim is True\n\n"
        "Question: What is 2 + 2?\n"
        "Claim: 2 + 2 = 5.\n"
        "I think this Claim is False\n\n"
    )

    target_style_1 = (
        f"Question: {question}\n"
        f"Claim: {claim}\n"
        "I think this Claim is "
    )

    target_style_2 = (
        f"Question: {question}\n"
        f"Claim: {claim}\n"
        "Is this claim true or false?\n"
        "Answer with one word only: True or False.\n"
        "Answer: "
    )

    return [
        shared_instruction + fewshot_block + target_style_1,
        shared_instruction + target_style_2,
    ]


def query_one(
    client: OpenAI,
    model: str,
    name: str,
    question: str,
    claim: str,
    max_retries: int,
    min_signal_mass: float,
) -> PromptResult:
    prompt_variants = build_prompt_variants(question, claim)
    last_result: PromptResult | None = None

    for attempt in range(1, max_retries + 1):
        prompt = prompt_variants[(attempt - 1) % len(prompt_variants)]
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=20,
            timeout=120,
        )
        choice = response.choices[0]
        completion = (choice.text or "").strip()
        top_logprobs = normalize_top_logprobs(choice)
        score_obj = score_true_false_from_top_logprobs(
            top_logprobs,
            min_signal_mass=min_signal_mass,
        )

        last_result = PromptResult(
            name=name,
            prompt=prompt,
            completion=completion,
            score=score_obj.score,
            predicted_label=score_obj.predicted_label_text,
            prob_true=score_obj.prob_true,
            prob_false=score_obj.prob_false,
            signal_mass=score_obj.signal_mass,
            valid_signal=score_obj.valid_signal,
            attempt=attempt,
            top_logprobs=top_logprobs,
        )

        if score_obj.valid_signal:
            return last_result

    if last_result is None:
        raise RuntimeError(f"No attempts executed for {name}.")
    return last_result


def default_prompts() -> list[tuple[str, str, str]]:
    return [
        (
            "math_true_claim",
            "What is 2 + 2?",
            "2 + 2 = 4.",
        ),
        (
            "math_false_claim",
            "What is 2 + 2?",
            "2 + 2 = 5.",
        ),
        (
            "factual_true_claim",
            "What is the capital of France?",
            "Paris is the capital of France.",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check True/False logprob scoring on base model.")
    parser.add_argument("--model", default=None, help="Base model ID (defaults to HYPERBOLIC_BASE_MODEL).")
    parser.add_argument("--base-url", default=None, help="API base URL (defaults to HYPERBOLIC_BASE_URL).")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for JSON artifact. Defaults to outputs/raw/logprob_sanity_<timestamp>.json",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum attempts per prompt before failing signal validation.",
    )
    parser.add_argument(
        "--min-signal-mass",
        type=float,
        default=DEFAULT_MIN_SIGNAL_MASS,
        help="Minimum combined probability mass on True/False tokens to mark a result as valid.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        print("Missing HYPERBOLIC_API_KEY in .env")
        return 2

    model = args.model or os.getenv("HYPERBOLIC_BASE_MODEL", "meta-llama/Meta-Llama-3.1-405B")
    base_url = args.base_url or os.getenv("HYPERBOLIC_BASE_URL", DEFAULT_BASE_URL)

    client = OpenAI(api_key=api_key, base_url=base_url)

    results: list[PromptResult] = []
    for name, question, claim in default_prompts():
        try:
            results.append(
                query_one(
                    client=client,
                    model=model,
                    name=name,
                    question=question,
                    claim=claim,
                    max_retries=args.max_retries,
                    min_signal_mass=args.min_signal_mass,
                )
            )
        except (APIStatusError, APIError, Exception) as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            return 1

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print("Results:")
    for result in results:
        print(f"- {result.name}")
        print(f"  completion={result.completion!r}")
        print(f"  score={result.score:.6f} (logP(True)-logP(False))")
        print(f"  prob_true={result.prob_true:.6f} prob_false={result.prob_false:.6f}")
        print(f"  signal_mass={result.signal_mass:.6f} valid_signal={result.valid_signal}")
        print(f"  predicted_label={result.predicted_label}")
        print(f"  attempts_used={result.attempt}")
        if result.top_logprobs:
            sorted_items = sorted(result.top_logprobs.items(), key=lambda kv: kv[1], reverse=True)[:8]
            tokens_preview = ", ".join([f"{tok!r}:{lp:.3f}" for tok, lp in sorted_items])
            print(f"  top_logprobs_preview={tokens_preview}")
        else:
            print("  top_logprobs_preview=<empty>")

    invalid = [result for result in results if not result.valid_signal]
    if invalid:
        print("\n[FAIL] Missing reliable True/False logprob signal after retries for:")
        for result in invalid:
            print(f"  - {result.name} (signal_mass={result.signal_mass:.6f}, attempts={result.attempt})")
        print("Adjust prompt style or use a model/endpoint that exposes stable True/False top logprobs.")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("outputs/raw") / f"logprob_sanity_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": timestamp,
        "model": model,
        "base_url": base_url,
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

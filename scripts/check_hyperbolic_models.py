#!/usr/bin/env python3
"""Verify Hyperbolic API credentials and model availability."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional

from dotenv import load_dotenv
from openai import APIError, APIStatusError, OpenAI


DEFAULT_BASE_URL = "https://api.hyperbolic.xyz/v1"


@dataclass
class AttemptResult:
    model: str
    endpoint: str
    ok: bool
    detail: str


def build_candidates(preferred: str, alternatives: Iterable[str], extras: Iterable[str]) -> list[str]:
    out = []
    for model in [preferred, *extras, *alternatives]:
        if model and model not in out:
            out.append(model)
    return out


def parse_csv_env(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def try_text_completion(client: OpenAI, model: str) -> AttemptResult:
    try:
        response = client.completions.create(
            model=model,
            prompt='Reply with only "OK".',
            max_tokens=6,
            temperature=0,
            logprobs=5,
            timeout=120,
        )
        text = (response.choices[0].text or "").strip()
        return AttemptResult(model=model, endpoint="completions", ok=True, detail=text)
    except (APIStatusError, APIError, Exception) as exc:
        return AttemptResult(
            model=model,
            endpoint="completions",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def try_chat_completion(client: OpenAI, model: str) -> AttemptResult:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with only "OK".'}],
            max_tokens=6,
            temperature=0,
            timeout=120,
        )
        text = (response.choices[0].message.content or "").strip()
        return AttemptResult(model=model, endpoint="chat.completions", ok=True, detail=text)
    except (APIStatusError, APIError, Exception) as exc:
        return AttemptResult(
            model=model,
            endpoint="chat.completions",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


def first_success(results: list[AttemptResult]) -> Optional[AttemptResult]:
    for result in results:
        if result.ok:
            return result
    return None


def print_attempts(title: str, results: list[AttemptResult]) -> None:
    print(f"\n{title}")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"  [{status}] {result.endpoint} :: {result.model}")
        print(f"         {result.detail}")


def run_checks(
    client: OpenAI, candidates: list[str], endpoint_order: list[str]
) -> list[AttemptResult]:
    attempts: list[AttemptResult] = []
    for candidate in candidates:
        for endpoint in endpoint_order:
            if endpoint == "chat.completions":
                result = try_chat_completion(client, candidate)
            elif endpoint == "completions":
                result = try_text_completion(client, candidate)
            else:
                attempts.append(
                    AttemptResult(
                        model=candidate,
                        endpoint=endpoint,
                        ok=False,
                        detail=f"Unknown endpoint '{endpoint}'",
                    )
                )
                continue
            attempts.append(result)
            if result.ok:
                return attempts
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Hyperbolic credentials + model availability for sprint models."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print loaded configuration only; do not call the API.",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail unless the exact models from HYPERBOLIC_BASE_MODEL and HYPERBOLIC_CHAT_MODEL pass.",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("HYPERBOLIC_API_KEY")
    base_url = os.getenv("HYPERBOLIC_BASE_URL", DEFAULT_BASE_URL)

    base_model = os.getenv("HYPERBOLIC_BASE_MODEL", "Llama-3.1-405B")
    chat_model = os.getenv("HYPERBOLIC_CHAT_MODEL", "Llama-3.1-405B-instruct")
    chat_fallbacks = parse_csv_env(os.getenv("HYPERBOLIC_CHAT_FALLBACK_MODELS"))
    require_exact = args.require_exact or parse_bool_env(
        os.getenv("HYPERBOLIC_REQUIRE_EXACT_MODELS"), default=False
    )

    base_endpoint_order = ["completions", "chat.completions"]
    chat_endpoint_order = ["chat.completions", "completions"]

    base_candidates = build_candidates(
        base_model,
        [
            "meta-llama/Meta-Llama-3.1-405B",
            "meta-llama/Llama-3.1-405B",
            "Llama-3.1-405B",
        ],
        [],
    )
    chat_candidates = build_candidates(
        chat_model,
        [
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct",
            "Llama-3.1-405B-Instruct",
            "Llama-3.1-405B-instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ],
        chat_fallbacks,
    )

    print("Hyperbolic verification config:")
    print(f"  base_url={base_url}")
    print(f"  base_candidates={base_candidates}")
    print(f"  chat_candidates={chat_candidates}")
    print(f"  base_endpoint_order={base_endpoint_order}")
    print(f"  chat_endpoint_order={chat_endpoint_order}")
    print(f"  require_exact={require_exact}")

    if args.dry_run:
        print("\nDry run complete.")
        return 0

    if not api_key:
        print("\nMissing HYPERBOLIC_API_KEY. Set it in .env (copy from .env.example).")
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"\nModel listing: Available ({len(model_ids)} models)")
        print("  First 20 model ids:")
        for mid in model_ids[:20]:
            print(f"    - {mid}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nModel listing failed (continuing): {type(exc).__name__}: {exc}")

    base_results = run_checks(client, base_candidates, base_endpoint_order)
    chat_results = run_checks(client, chat_candidates, chat_endpoint_order)

    print_attempts("Base model check (prefers completions)", base_results)
    print_attempts("Chat model check (prefers chat.completions)", chat_results)

    base_ok = first_success(base_results)
    chat_ok = first_success(chat_results)

    print("\nSummary")
    print(f"  Base model: {'PASS' if base_ok else 'FAIL'}")
    if base_ok:
        print(f"    selected_model={base_ok.model}")
        print(f"    selected_endpoint={base_ok.endpoint}")
    print(f"  Chat model: {'PASS' if chat_ok else 'FAIL'}")
    if chat_ok:
        print(f"    selected_model={chat_ok.model}")
        print(f"    selected_endpoint={chat_ok.endpoint}")

    if not (base_ok and chat_ok):
        return 1

    if require_exact:
        exact_ok = (base_ok.model == base_model) and (chat_ok.model == chat_model)
        if not exact_ok:
            print("\nStrict mode failure: fallback model was used.")
            print(f"  requested_base={base_model} selected_base={base_ok.model}")
            print(f"  requested_chat={chat_model} selected_chat={chat_ok.model}")
            return 1

    if base_ok.model != base_model or chat_ok.model != chat_model:
        print("\nNote: fallback model selected for at least one check.")
        print(f"  requested_base={base_model} selected_base={base_ok.model}")
        print(f"  requested_chat={chat_model} selected_chat={chat_ok.model}")
        print("  This is acceptable for local development but not for final sprint replication.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

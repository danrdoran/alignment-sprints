"""Prompt builders for TruthfulQA ICM experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

from .data import TruthfulQAExample


TRUTHFULQA_TEMPLATE = "Question: {question}\nClaim: {claim}\nI think this Claim is "


def label_to_text(label: int | bool) -> str:
    """Map binary label to True/False text."""
    return "True" if int(label) == 1 else "False"


def build_claim_prompt(question: str, claim: str) -> str:
    """Build a single unlabeled claim prompt in paper format."""
    return TRUTHFULQA_TEMPLATE.format(question=question, claim=claim)


def build_example_prompt(example: TruthfulQAExample) -> str:
    """Build the unlabeled prompt for a dataset example."""
    return build_claim_prompt(example.question, example.choice)


def build_labeled_example_line(example: TruthfulQAExample, label: int | bool) -> str:
    """Build one demonstration line ending with explicit True/False label."""
    return build_example_prompt(example) + label_to_text(label)


def build_fewshot_prompt(
    target: TruthfulQAExample,
    demonstrations: Sequence[tuple[TruthfulQAExample, int | bool]],
) -> str:
    """
    Build few-shot prompt:
    - each demonstration is a full prompt plus label
    - target is unlabeled prompt at the end
    """
    blocks: list[str] = []
    for demo_example, demo_label in demonstrations:
        blocks.append(build_labeled_example_line(demo_example, demo_label))
    blocks.append(build_example_prompt(target))
    return "\n\n".join(blocks)


def build_fewshot_prompt_from_examples(
    target: TruthfulQAExample,
    demonstration_examples: Iterable[TruthfulQAExample],
) -> str:
    """Convenience wrapper using each demo example's own label field."""
    demos = [(demo, demo.label) for demo in demonstration_examples]
    return build_fewshot_prompt(target=target, demonstrations=demos)

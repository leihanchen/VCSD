#!/usr/bin/env python3
"""Exact-match reward for multiple-choice answer-letter tasks."""

from __future__ import annotations

import re
from typing import Any


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _strip_reasoning(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def _extract_answer_span(text: str) -> str:
    text = _strip_reasoning(_as_text(text))
    if not text:
        return ""

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(
        r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is)?\s*:?\s*[\(\[]?\s*([A-F])\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    return text.strip()


def extract_mcq_option(text: Any) -> str:
    """Extract the first answer option letter A-F from text."""
    answer = _extract_answer_span(_as_text(text))
    if not answer:
        return ""

    match = re.match(r"^\s*[\(\[]?\s*([A-F])(?:\s*[\)\].:\-]|(?:\s+|$))", answer, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b(?:option|choice|letter)\s*[\(\[]?\s*([A-F])\b", answer, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:^|[\n\r])\s*[\(\[]?\s*([A-F])(?:\s*[\)\].:\-]|(?:\s+|$))", answer, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def _ground_truth_option(ground_truth: Any, extra_info: dict[str, Any]) -> str:
    gt = extract_mcq_option(ground_truth)
    if gt:
        return gt
    return extract_mcq_option(extra_info.get("answer"))


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return 1.0 iff the generated option letter exactly matches the label."""
    del data_source
    extra_info = extra_info or {}
    pred = extract_mcq_option(solution_str)
    gt = _ground_truth_option(ground_truth, extra_info)
    is_correct = bool(pred and gt and pred == gt)

    return {
        "score": float(is_correct),
        "acc": float(is_correct),
        "format_ok": float(bool(pred)),
        "pred": pred,
        "gt": gt,
        "judge_source": "mcq_exact",
    }


def _self_test() -> None:
    cases = [
        ("D", "D", 1.0),
        ("Answer: D", "D", 1.0),
        ("<answer>D</answer>", "D", 1.0),
        ("The correct answer is D because ...", "D", 1.0),
        ("Therefore, the correct answer is:\n\nC. volleyball", "C", 1.0),
        ("D. purple", "D", 1.0),
        ("The answer is C", "D", 0.0),
        ("I cannot tell", "D", 0.0),
        ("This is a long answer without a final option.", "A", 0.0),
    ]
    for response, gt, expected in cases:
        actual = compute_score(solution_str=response, ground_truth=gt)["acc"]
        assert actual == expected, (response, gt, expected, actual)


if __name__ == "__main__":
    _self_test()
    print("mcq_exact_reward self-test passed")

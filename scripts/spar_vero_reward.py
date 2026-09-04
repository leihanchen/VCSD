#!/usr/bin/env python3
"""SpatialLadder-aligned rewards for cvis-tmu/spar-vero-rl."""

from __future__ import annotations

import re
from typing import Any


_NUMBER_PATTERN = re.compile(r"(?<!\^)−?-?\d+(?:,\d{3})*(?:\.\d+)?")
_MULTIVIEW_NUMERIC_TYPES = {
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oc_mv",
    "distance_prediction_oo_mv",
}


def _answer_segment(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    matches = list(
        re.finditer(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    )
    return matches[-1].group(1).strip() if matches else text


def _mcq_option(value: Any) -> str:
    answer = _answer_segment(value)
    match = re.match(r"^\s*[\(\[]?\s*([A-F])(?:\s*[\)\].:\-]|\s*$)", answer, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:option|choice|letter)\s*[\(\[]?\s*([A-F])\b", answer, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _numbers(value: Any) -> list[float]:
    return [float(match.replace("−", "-").replace(",", "")) for match in _NUMBER_PATTERN.findall(str(value))]


def _mean_relative_accuracy(prediction: float, target: float) -> float:
    error = abs(prediction - target) if target == 0.0 else abs((prediction - target) / target)
    # Match SpatialLadder's 11-point np.linspace(0.5, 0.95, 11) implementation.
    confidences = [0.5 + index * (0.95 - 0.5) / 10 for index in range(11)]
    return sum(error <= 1.0 - confidence for confidence in confidences) / len(confidences)


def _parse_movement(value: Any) -> dict[str, float]:
    movement: dict[str, float] = {}
    for item in _answer_segment(value).split(","):
        key, raw_value = item.strip().split(":", 1)
        movement[key.strip()] = float(raw_value.strip())
    return movement


def _movement_axes(movement: dict[str, float]) -> list[float]:
    return [
        movement.get("move_right", 0.0) - movement.get("move_left", 0.0),
        movement.get("move_up", 0.0) - movement.get("move_down", 0.0),
        movement.get("move_forward", 0.0)
        - movement.get("move_backward", movement.get("move_back", 0.0)),
        movement.get("rotate_right", 0.0) - movement.get("rotate_left", 0.0),
        movement.get("rotate_up", 0.0) - movement.get("rotate_down", 0.0),
    ]


def _normalized_string(value: Any) -> str:
    return " ".join(_answer_segment(value).casefold().split())


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Score one SPAR-Vero response using its declared reward type."""
    del data_source
    extra_info = extra_info or {}
    reward_type = str(extra_info.get("reward_type", "")).strip().lower()
    question_type = str(extra_info.get("type", "")).strip()
    result: dict[str, Any] = {
        "score": 0.0,
        "acc": 0.0,
        "pred": _answer_segment(solution_str),
        "gt": "" if ground_truth is None else str(ground_truth),
        "reward_type": reward_type,
        "judge_source": "spar_invalid",
    }

    try:
        if reward_type == "multiple_choice":
            pred = _mcq_option(solution_str)
            gt = _mcq_option(ground_truth)
            score = float(bool(pred and gt and pred == gt))
            result.update(score=score, acc=score, pred=pred, gt=gt, judge_source="spar_multiple_choice")
        elif reward_type == "numeric":
            pred_numbers = _numbers(_answer_segment(solution_str))
            gt_numbers = _numbers(ground_truth)
            if pred_numbers and gt_numbers:
                pred = pred_numbers[-1] if question_type in _MULTIVIEW_NUMERIC_TYPES else pred_numbers[0]
                gt = gt_numbers[0]
                score = _mean_relative_accuracy(pred, gt)
                result.update(score=score, acc=score, pred=pred, gt=gt, judge_source="spar_numeric_mra")
        elif reward_type == "string_match" and question_type == "view_change_infer":
            pred_axes = _movement_axes(_parse_movement(solution_str))
            gt_axes = _movement_axes(_parse_movement(ground_truth))
            # Preserve SpatialLadder's argument order for the view-change metric.
            score = sum(_mean_relative_accuracy(gt, pred) for gt, pred in zip(gt_axes, pred_axes, strict=True)) / 5
            result.update(score=score, acc=score, pred=pred_axes, gt=gt_axes, judge_source="spar_view_change_mra")
        elif reward_type == "string_match":
            pred = _normalized_string(solution_str)
            gt = _normalized_string(ground_truth)
            score = float(bool(pred and gt and pred == gt))
            result.update(score=score, acc=score, pred=pred, gt=gt, judge_source="spar_string_match")
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    return result

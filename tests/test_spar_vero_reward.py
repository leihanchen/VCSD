import pytest

from scripts.spar_vero_reward import compute_score


def test_multiple_choice_uses_the_final_answer_letter() -> None:
    result = compute_score(
        solution_str="<think>A is tempting.</think><answer>B</answer>",
        ground_truth="B",
        extra_info={"reward_type": "multiple_choice", "type": "depth_prediction_oc"},
    )

    assert result["score"] == 1.0
    assert result["pred"] == "B"
    assert result["judge_source"] == "spar_multiple_choice"


def test_numeric_uses_mean_relative_accuracy() -> None:
    exact = compute_score(
        solution_str="1.5",
        ground_truth="1.5",
        extra_info={"reward_type": "numeric", "type": "distance_prediction_oc"},
    )
    partial = compute_score(
        solution_str="2.1",
        ground_truth="1.5",
        extra_info={"reward_type": "numeric", "type": "distance_prediction_oc"},
    )

    assert exact["score"] == 1.0
    assert partial["score"] == pytest.approx(3 / 11)
    assert partial["judge_source"] == "spar_numeric_mra"


def test_view_change_uses_structured_five_axis_metric() -> None:
    answer = "move_left:0,move_down:0,move_forward:0,rotate_up:10,rotate_right:15"
    exact = compute_score(
        solution_str=answer,
        ground_truth=answer,
        extra_info={"reward_type": "string_match", "type": "view_change_infer"},
    )
    malformed = compute_score(
        solution_str="move_left:not-a-number",
        ground_truth=answer,
        extra_info={"reward_type": "string_match", "type": "view_change_infer"},
    )

    assert exact["score"] == 1.0
    assert exact["judge_source"] == "spar_view_change_mra"
    assert malformed["score"] == 0.0


def test_other_strings_are_case_and_whitespace_normalized() -> None:
    result = compute_score(
        solution_str="  Object1   (blue point) ",
        ground_truth="object1 (blue point)",
        extra_info={"reward_type": "string_match", "type": "distance_infer_center_oo_mv"},
    )

    assert result["score"] == 1.0
    assert result["judge_source"] == "spar_string_match"

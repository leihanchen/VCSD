import re
import string
from decimal import Decimal, InvalidOperation


def _extract_answer_segment(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    matches = list(
        re.finditer(
            r"(?:final\s+answer|answer)\s*[:：]\s*(.+?)(?:\n|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if matches:
        return matches[-1].group(1).strip()

    return text


def _extract_mcq_option(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    segment = _extract_answer_segment(text)
    candidates = []
    for source in (segment, text):
        candidates.extend(re.findall(r"\(([A-Fa-f])\)", source))
        candidates.extend(re.findall(r"(?<![A-Za-z])([A-Fa-f])(?![A-Za-z])", source))
        if candidates:
            return candidates[-1].upper()
    return ""


def _format_reward(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    stripped = text.strip()
    if re.search(r"<answer>\s*[^<>\n]+\s*</answer>", stripped, flags=re.IGNORECASE):
        return 1.0
    if re.search(r"(?:final\s+answer|answer)\s*[:：]\s*[A-Fa-f]\b", stripped, flags=re.IGNORECASE):
        return 1.0
    if re.fullmatch(r"\s*[A-Fa-f]\s*", stripped):
        return 1.0
    return 0.0


def _normalize_text(text: str) -> str:
    lowered = str(text).lower().strip()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(no_punc.split())


def _parse_number(text: str) -> Decimal | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace("−", "-"))
    if not match:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_weight: float = 0.0,
    **kwargs,
):
    """Rule reward for Vision-OPD bbox VQA.

    The default score is accuracy-only. Set format_weight=0.05 or 0.1 from
    Hydra overrides to add a small formatting bonus without changing the
    correctness decision.
    """

    gt = str(ground_truth).strip()
    answer_segment = _extract_answer_segment(solution_str)
    fmt = _format_reward(solution_str)

    if re.fullmatch(r"[A-Fa-f]", gt):
        pred = _extract_mcq_option(solution_str)
        acc = float(bool(pred) and pred.upper() == gt.upper())
        return {
            "score": acc + float(format_weight) * fmt,
            "acc": acc,
            "format": fmt,
            "pred": pred,
            "extracted_answer": answer_segment,
            "judge_source": "vision_opd_mcq_rule",
        }

    pred_num = _parse_number(answer_segment)
    gt_num = _parse_number(gt)
    if pred_num is not None and gt_num is not None:
        acc = float(pred_num == gt_num)
        return {
            "score": acc + float(format_weight) * fmt,
            "acc": acc,
            "format": fmt,
            "pred": str(pred_num),
            "extracted_answer": answer_segment,
            "judge_source": "vision_opd_numeric_rule",
        }

    acc = float(_normalize_text(answer_segment) == _normalize_text(gt))
    return {
        "score": acc + float(format_weight) * fmt,
        "acc": acc,
        "format": fmt,
        "pred": answer_segment,
        "extracted_answer": answer_segment,
        "judge_source": "vision_opd_normalized_em",
    }

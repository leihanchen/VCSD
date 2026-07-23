# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""VCSD contrast-sharpened self-distillation loss.

This is the final, minimal version of the method: a **contrast-sharpened distillation
target** built from the teacher's image-present vs. control (image-removed / blacked-out)
full-vocab log-probs, distilled into the student with a plain token-averaged KL over the
response mask (no reliability-aware token weighting, no gating).
"""

from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


def token_divergence(
    teacher_all_log_probs: torch.Tensor,
    student_all_log_probs: torch.Tensor,
    *,
    alpha: float = 0.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-token full-vocab divergence: 0=forward KL, 1=reverse KL, intermediate=generalized JSD."""
    alpha = float(alpha)
    if alpha == 0.0:
        loss = F.kl_div(student_all_log_probs, teacher_all_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        loss = F.kl_div(teacher_all_log_probs, student_all_log_probs, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=student_all_log_probs.dtype, device=student_all_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_all_log_probs + torch.log1p(-alpha_t),
                    teacher_all_log_probs + torch.log(alpha_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_all_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_all_log_probs, reduction="none", log_target=True)
        loss = torch.lerp(kl_student, kl_teacher, alpha_t)
    return loss.sum(dim=-1) * (float(temperature) ** 2)


def token_kl(
    teacher_all_log_probs: torch.Tensor,
    student_all_log_probs: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-token forward KL KL(p_teacher || p_student) over the full vocabulary."""
    return token_divergence(
        teacher_all_log_probs=teacher_all_log_probs,
        student_all_log_probs=student_all_log_probs,
        alpha=0.0,
        temperature=temperature,
    )


@torch.no_grad()
def build_contrast_target(
    teacher_hi_all_log_probs: torch.Tensor,
    teacher_ctrl_all_log_probs: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
    anchor_coef: float = 1.0,
    exclude_token_ids: Optional[Sequence[int]] = None,
    metric_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Contrast-sharpened distillation target: softmax(anchor_coef·lp_hi + α (lp_hi − lp_ctrl))
    restricted to the plausibility set {w : p_hi(w) ≥ β · max p_hi}.

    anchor_coef=1 (default) keeps the lp_hi anchor; anchor_coef=0 + α=1 gives
    softmax(lp_hi − lp_ctrl), the pure contrastive-decoding target.

    Safety properties:
      - Always a valid distribution (softmax of a real vector).
      - Support ⊆ hi's plausibility set: mass can only be redistributed among tokens the hi
        teacher already considers plausible (log p_hi within log β of the max) — the student
        can never be taught a token the plain teacher wouldn't produce.
      - ``exclude_token_ids`` (e.g. <|im_end|>/<|endoftext|>): the tilt is disabled for these
        vocab entries (they keep their plain lp_hi score), so the tilt cannot boost
        early-termination tokens and silently shorten responses.

    Inputs are temperature-softened log-probs (as produced upstream); output is log-probs in the
    same convention, detached.
    """
    lp_hi = teacher_hi_all_log_probs
    lp_ctrl = teacher_ctrl_all_log_probs
    tilted = float(anchor_coef) * lp_hi + float(alpha) * (lp_hi - lp_ctrl)

    if exclude_token_ids:
        idx = torch.tensor(list(exclude_token_ids), device=lp_hi.device, dtype=torch.long)
        tilted.index_copy_(-1, idx, lp_hi.index_select(-1, idx))

    max_lp = lp_hi.max(dim=-1, keepdim=True).values
    plausible = lp_hi >= (max_lp + torch.log(torch.tensor(float(beta), device=lp_hi.device)))
    tilted = tilted.masked_fill(~plausible, float("-inf"))
    # Clamp the -inf log-probs of masked-out tokens to a large finite negative: exp(-1e4) is
    # exactly 0.0 in fp32 (numerically identical distribution) but avoids the 0 * (-inf) = NaN
    # that F.kl_div(..., log_target=True) produces on true -inf entries.
    target = torch.log_softmax(tilted, dim=-1).clamp_min(-1e4)

    # Drift monitoring: KL(target || p_hi) and argmax-change rate, masked to real (non-padding)
    # positions (padded rows carry all-zero lp_hi vectors that are not valid log-distributions).
    kl_vs_hi = F.kl_div(lp_hi, target, reduction="none", log_target=True).sum(-1)
    argmax_changed = (target.argmax(-1) != lp_hi.argmax(-1)).float()
    if metric_mask is not None:
        m = metric_mask.to(dtype=kl_vs_hi.dtype)
        denom = m.sum().clamp(min=1.0)
        kl_mean = (kl_vs_hi * m).sum() / denom
        change_rate = (argmax_changed * m).sum() / denom
    else:
        kl_mean = kl_vs_hi.mean()
        change_rate = argmax_changed.mean()
    metrics = {
        "vcsd/contrast_target_kl_vs_hi": kl_mean.item(),
        "vcsd/contrast_argmax_change_rate": change_rate.item(),
    }
    return target.detach(), metrics


def vcsd_kd_loss(
    student_all_log_probs: torch.Tensor,
    teacher_all_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 2.0,
    divergence_alpha: float = 0.0,
    target_mode: str = "contrast",
    teacher_ctrl_all_log_probs: Optional[torch.Tensor] = None,
    contrast_alpha: float = 1.0,
    contrast_beta: float = 0.1,
    contrast_anchor_coef: float = 1.0,
    contrast_exclude_token_ids: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-averaged full-vocab KD toward the contrast-sharpened (or plain teacher) target.

    The loss is a plain mean of the per-token divergence over the response mask (optionally
    intersected with ``self_distillation_mask``) — no per-token reliability weighting, no gating.
    """
    loss_mask = response_mask.to(dtype=student_all_log_probs.dtype)
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1).to(dtype=loss_mask.dtype)

    contrast_metrics: dict[str, Any] = {}
    if target_mode == "contrast":
        if teacher_ctrl_all_log_probs is None:
            raise ValueError("target_mode='contrast' requires teacher_ctrl_all_log_probs (full-vocab ctrl log-probs).")
        teacher_all_log_probs, contrast_metrics = build_contrast_target(
            teacher_hi_all_log_probs=teacher_all_log_probs,
            teacher_ctrl_all_log_probs=teacher_ctrl_all_log_probs,
            alpha=contrast_alpha,
            beta=contrast_beta,
            anchor_coef=contrast_anchor_coef,
            exclude_token_ids=contrast_exclude_token_ids,
            metric_mask=loss_mask,
        )
    elif target_mode != "teacher":
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    per_token_kl = token_divergence(
        teacher_all_log_probs=teacher_all_log_probs,
        student_all_log_probs=student_all_log_probs,
        alpha=divergence_alpha,
        temperature=temperature,
    )

    denom = loss_mask.sum().clamp(min=1.0)
    loss = (per_token_kl * loss_mask).sum() / denom

    metrics = {
        "vcsd/kd_loss": loss.detach().item(),
        "vcsd/token_kl_mean": loss.detach().item(),
        "vcsd/num_tokens": loss_mask.sum().detach().item(),
        "vcsd/divergence_alpha": float(divergence_alpha),
    }
    metrics.update(contrast_metrics)
    return loss, metrics

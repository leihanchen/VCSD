#!/bin/bash
# VCSD launcher: contrast-sharpened self-distillation, standard config.
#
#   contrast target = softmax(lp_hi + alpha*(lp_hi - lp_ctrl)) within hi's beta-plausibility set,
#   alpha=1.0, beta=0.1, control = black image, termination tokens exempt from the tilt.
#
# Usage:
#   bash scripts/run_experiment_contrast_standard.sh
set -euo pipefail

export EXPERIMENT="${EXPERIMENT:-black}"
export MODEL_SIZE="${MODEL_SIZE:-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

exec bash "$(dirname "$0")/run_vcsd.sh" \
    actor_rollout_ref.actor.self_distillation.vcsd_target_mode=contrast \
    actor_rollout_ref.actor.self_distillation.vcsd_contrast_alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.vcsd_contrast_beta=0.1 \
    'actor_rollout_ref.actor.self_distillation.vcsd_contrast_exclude_token_ids=[151643,151645]' \
    "$@"

#!/bin/bash
# VCSD contrast-standard launcher using the cvis-tmu/spar-vero-rl training set.
#
# The dataset splits are downloaded and converted to local Parquet only when this
# script is run. Referenced SPAR-7M images must already exist under SPAR_IMAGE_ROOT.
#
# Usage:
#   bash scripts/run_experiment_contrast_standard_spar_vero_rl.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DATASET_ID="${DATASET_ID:-cvis-tmu/spar-vero-rl}"
DATASET_TRAIN_SPLIT="${DATASET_TRAIN_SPLIT:-${DATASET_SPLIT:-train}}"
DATASET_VAL_SPLIT="${DATASET_VAL_SPLIT:-test}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/spar-vero-rl}"
TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${DATASET_DIR}/${DATASET_TRAIN_SPLIT}.parquet}"
TASK_VAL_FILE="${TASK_VAL_FILE:-${DATASET_DIR}/${DATASET_VAL_SPLIT}.parquet}"
SPAR_IMAGE_ROOT="${SPAR_IMAGE_ROOT:-${PROJECT_ROOT}/data}"
SPAR_VAL_ENABLE="${SPAR_VAL_ENABLE:-1}"
SPAR_VAL_BATCH_SIZE="${SPAR_VAL_BATCH_SIZE:-16}"
SPAR_VAL_TEST_FREQ="${SPAR_VAL_TEST_FREQ:-10}"
SPAR_VAL_BEFORE_TRAIN="${SPAR_VAL_BEFORE_TRAIN:-True}"
SPAR_DATASET_CLASS="pkg://verl/utils/dataset/spar_vero_rl_dataset"
SPAR_REWARD_FILE="${PROJECT_ROOT}/scripts/spar_vero_reward.py"

if [[ ! -d "${SPAR_IMAGE_ROOT}/SPAR-7M" ]]; then
    echo "ERROR: SPAR-7M image tree not found: ${SPAR_IMAGE_ROOT}/SPAR-7M" >&2
    echo "       Set SPAR_IMAGE_ROOT to the directory containing SPAR-7M/." >&2
    exit 1
fi

prepare_split() {
    local split="$1"
    local output_file="$2"
    [[ -f "$output_file" ]] && return

    mkdir -p "$(dirname "$output_file")"
    echo "Preparing Hugging Face dataset ${DATASET_ID} (${split}) at ${output_file}"
    DATASET_ID="$DATASET_ID" \
    DATASET_SPLIT="$split" \
    DATASET_OUTPUT_FILE="$output_file" \
    HF_HUB_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    python3 - <<'PY'
import os
from pathlib import Path

from datasets import load_dataset

dataset = load_dataset(os.environ["DATASET_ID"], split=os.environ["DATASET_SPLIT"])
output = Path(os.environ["DATASET_OUTPUT_FILE"])
temporary_output = output.with_suffix(output.suffix + ".incomplete")
try:
    dataset.to_parquet(temporary_output)
    temporary_output.replace(output)
finally:
    temporary_output.unlink(missing_ok=True)
PY
}

prepare_split "$DATASET_TRAIN_SPLIT" "$TASK_TRAIN_FILE"

export TASK_TRAIN_FILE
# Disable the base launcher's answer-letter split; SPAR uses its own test split
# and mixed-type reward function.
export ANSWER_VAL_ENABLE=0

SPAR_ARGS=(
    data.custom_cls.path="$SPAR_DATASET_CLASS"
    data.custom_cls.name=SparVeroRLDataset
    +data.image_root="$SPAR_IMAGE_ROOT"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers="${SPAR_FILTER_WORKERS:-4}"
)

if [[ "$SPAR_VAL_ENABLE" == "1" ]]; then
    prepare_split "$DATASET_VAL_SPLIT" "$TASK_VAL_FILE"
    SPAR_ARGS+=(
        data.val_files="[\"$TASK_VAL_FILE\"]"
        data.val_batch_size="$SPAR_VAL_BATCH_SIZE"
        custom_reward_function.path="$SPAR_REWARD_FILE"
        custom_reward_function.name=compute_score
        trainer.val_before_train="$SPAR_VAL_BEFORE_TRAIN"
        trainer.test_freq="$SPAR_VAL_TEST_FREQ"
        actor_rollout_ref.rollout.val_kwargs.n=1
    )
fi

exec bash "${PROJECT_ROOT}/scripts/run_experiment_contrast_standard.sh" "${SPAR_ARGS[@]}" "$@"

#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED_PROJECT_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
CONFIG_NAME="vopd"
# Model size: 4B (default) or 2B. Override with MODEL_SIZE=2B, or set MODEL_PATH directly.
MODEL_SIZE="${MODEL_SIZE:-4B}"
if [[ "$MODEL_SIZE" == "2B" ]]; then
    DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
else
    DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
fi
if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_REPO_DIR="${DEFAULT_MODEL_PATH//\//--}"
    # Search candidate HF cache dirs in priority order; pick the FIRST snapshot dir
    # that actually contains config.json (a complete download). This avoids using a
    # partially-downloaded snapshot (e.g. one with only model.safetensors and no config).
    CANDIDATE_CACHE_DIRS=(
        "${HOME}/.cache/huggingface/hub"
        "${HF_HOME:-${HOME}/.cache/huggingface}/hub"
        "${SHARED_PROJECT_ROOT}/cache/hub"
        "${PROJECT_ROOT}/cache/hub"
    )
    MODEL_PATH=""
    for cache_dir in "${CANDIDATE_CACHE_DIRS[@]}"; do
        snap_root="${cache_dir}/models--${MODEL_REPO_DIR}/snapshots"
        [[ -d "$snap_root" ]] || continue
        while IFS= read -r snap_dir; do
            has_weights=0
            if compgen -G "${snap_dir}/model*.safetensors" >/dev/null; then
                has_weights=1
            elif [[ -f "${snap_dir}/model.safetensors.index.json" || -f "${snap_dir}/pytorch_model.bin" ]]; then
                has_weights=1
            fi
            if [[ -f "${snap_dir}/config.json" && -f "${snap_dir}/tokenizer_config.json" && "$has_weights" == "1" ]]; then
                MODEL_PATH="$snap_dir"
                break
            fi
        done < <(find "$snap_root" -mindepth 1 -maxdepth 1 -type d | sort)
        [[ -n "$MODEL_PATH" ]] && break
    done
    if [[ -z "$MODEL_PATH" ]]; then
        echo "ERROR: no complete local snapshot for ${DEFAULT_MODEL_PATH} (config.json missing)." >&2
        echo "       Searched: ${CANDIDATE_CACHE_DIRS[*]}" >&2
        echo "       Re-download to a writable HF_HOME, or set MODEL_PATH explicitly." >&2
        exit 1
    fi
fi

EXPERIMENT="${EXPERIMENT:-black}"
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.05

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-96}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
LR="${LR:-2e-6}"
DONT_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ACTOR_USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU=$TRAIN_MAX_MODEL_LEN
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
DETECTED_GPUS="$(python3 - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-$DETECTED_GPUS}"
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
# Keep 10 (not 2): =2 deletes actor weights from all but the last 2 checkpoints
# *during* training, which turns every earlier global_step_N into a weightless
# data.pt shell and silently defeats the post-training prune_checkpoints.sh
# retention policy. Cost us all early checkpoints of three 437-step runs when
# we needed them to bisect a mode-collapse onset (2026-07-13). 10 covers most
# retrospective needs at ~270GB peak for a 2B run; disk is reclaimed by
# prune_checkpoints.sh right after training per the standing policy.
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-10}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"tensorboard\"]}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-8}"
CUSTOM_CHAT_TEMPLATE_FILE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"

DATA_DIR="${PROJECT_ROOT}/data"
TASK_TRAIN_FILE_WAS_DEFAULT=0
if [[ -z "${TASK_TRAIN_FILE:-}" ]]; then
    TASK_TRAIN_FILE_WAS_DEFAULT=1
    if [[ "$EXPERIMENT" == "degrade" ]]; then
        TASK_TRAIN_FILE="${DATA_DIR}/train_degraded.parquet"
    else
        TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"
    fi
fi

# degrade variant requires the offline-degraded parquet (run scripts/prepare_degraded_images.py first)
if [[ "$EXPERIMENT" == "degrade" && ! -f "$TASK_TRAIN_FILE" ]]; then
    echo "ERROR: degrade experiment needs $TASK_TRAIN_FILE." >&2
    echo "       Generate it first:  python3 scripts/prepare_degraded_images.py --input data/train.parquet --output data/train_degraded.parquet" >&2
    exit 1
fi

# --- Answer-letter validation split ------------------------------------------
# Enabled by default for these answer-letter VQA experiments. It keeps a fixed
# validation subset out of training and evaluates it with a lightweight exact-
# match reward. Disable with ANSWER_VAL_ENABLE=0.
ANSWER_VAL_ENABLE="${ANSWER_VAL_ENABLE:-1}"
ANSWER_VAL_SIZE="${ANSWER_VAL_SIZE:-256}"
ANSWER_VAL_SEED="${ANSWER_VAL_SEED:-42}"
ANSWER_VAL_BATCH_SIZE="${ANSWER_VAL_BATCH_SIZE:-16}"
ANSWER_VAL_TEST_FREQ="${ANSWER_VAL_TEST_FREQ:-10}"
ANSWER_VAL_BEFORE_TRAIN="${ANSWER_VAL_BEFORE_TRAIN:-True}"
ANSWER_VAL_TRAIN_FILE="${ANSWER_VAL_TRAIN_FILE:-${DATA_DIR}/train_answer.parquet}"
ANSWER_VAL_FILE="${ANSWER_VAL_FILE:-${DATA_DIR}/val_answer.parquet}"
ANSWER_VAL_REWARD_FILE="${ANSWER_VAL_REWARD_FILE:-${PROJECT_ROOT}/scripts/mcq_exact_reward.py}"
ANSWER_VAL_ARGS=()

if [[ "$ANSWER_VAL_ENABLE" == "1" ]]; then
    if [[ ! -f "$ANSWER_VAL_REWARD_FILE" ]]; then
        echo "ERROR: answer validation reward file not found: $ANSWER_VAL_REWARD_FILE" >&2
        exit 1
    fi

    if [[ ! -f "$ANSWER_VAL_TRAIN_FILE" || ! -f "$ANSWER_VAL_FILE" || "${DATA_DIR}/train.parquet" -nt "$ANSWER_VAL_TRAIN_FILE" || "${DATA_DIR}/train.parquet" -nt "$ANSWER_VAL_FILE" ]]; then
        python3 "${PROJECT_ROOT}/scripts/prepare_answer_val_split.py" \
            --input "${DATA_DIR}/train.parquet" \
            --train-output "$ANSWER_VAL_TRAIN_FILE" \
            --val-output "$ANSWER_VAL_FILE" \
            --val-size "$ANSWER_VAL_SIZE" \
            --seed "$ANSWER_VAL_SEED"
    fi

    if [[ "$TASK_TRAIN_FILE_WAS_DEFAULT" == "1" ]]; then
        if [[ "$EXPERIMENT" == "degrade" ]]; then
            ANSWER_VAL_DEGRADED_TRAIN_FILE="${ANSWER_VAL_DEGRADED_TRAIN_FILE:-${DATA_DIR}/train_degraded_answer.parquet}"
            ANSWER_VAL_DEGRADED_VAL_FILE="${ANSWER_VAL_DEGRADED_VAL_FILE:-${DATA_DIR}/val_degraded_answer.parquet}"
            if [[ ! -f "${DATA_DIR}/train_degraded.parquet" ]]; then
                echo "ERROR: answer validation for degrade needs ${DATA_DIR}/train_degraded.parquet." >&2
                echo "       Generate it first:  python3 scripts/prepare_degraded_images.py --input data/train.parquet --output data/train_degraded.parquet" >&2
                exit 1
            fi
            if [[ ! -f "$ANSWER_VAL_DEGRADED_TRAIN_FILE" || ! -f "$ANSWER_VAL_DEGRADED_VAL_FILE" || "${DATA_DIR}/train_degraded.parquet" -nt "$ANSWER_VAL_DEGRADED_TRAIN_FILE" || "${DATA_DIR}/train_degraded.parquet" -nt "$ANSWER_VAL_DEGRADED_VAL_FILE" ]]; then
                python3 "${PROJECT_ROOT}/scripts/prepare_answer_val_split.py" \
                    --input "${DATA_DIR}/train_degraded.parquet" \
                    --train-output "$ANSWER_VAL_DEGRADED_TRAIN_FILE" \
                    --val-output "$ANSWER_VAL_DEGRADED_VAL_FILE" \
                    --val-size "$ANSWER_VAL_SIZE" \
                    --seed "$ANSWER_VAL_SEED"
            fi
            TASK_TRAIN_FILE="$ANSWER_VAL_DEGRADED_TRAIN_FILE"
        else
            TASK_TRAIN_FILE="$ANSWER_VAL_TRAIN_FILE"
        fi
    fi

    ANSWER_VAL_ARGS+=(
        data.val_files="[\"$ANSWER_VAL_FILE\"]"
        data.val_batch_size="$ANSWER_VAL_BATCH_SIZE"
        custom_reward_function.path="$ANSWER_VAL_REWARD_FILE"
        custom_reward_function.name=compute_score
        trainer.val_before_train="$ANSWER_VAL_BEFORE_TRAIN"
        trainer.test_freq="$ANSWER_VAL_TEST_FREQ"
        actor_rollout_ref.rollout.val_kwargs.n=1
    )
fi

MODEL_NAME=$(basename "$DEFAULT_MODEL_PATH")
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-${EXPERIMENT}-${MODEL_NAME}}"
PROJECT_NAME="${PROJECT_NAME:-Vision-OPD}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
TRAINER_ROLLOUT_DATA_DIR="${TRAINER_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")
# VCSD requires full-vocab (non-top-k) distillation so student/teacher
# emit `all_logps` for the full-vocab KL. Set explicitly (do not rely on the default).
VCSD_FULL_LOGIT_ARG="actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
EXPERIMENT_ARGS=()
case "$EXPERIMENT" in
    visionopd)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.vcsd=False
            actor_rollout_ref.actor.self_distillation.distillation_topk=100
            actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
            actor_rollout_ref.actor.self_distillation.is_clip=2.0
        )
        ;;
    baseline)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=answer_hint
            actor_rollout_ref.actor.self_distillation.vcsd=False
            actor_rollout_ref.actor.self_distillation.distillation_topk=100
            actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
            actor_rollout_ref.actor.self_distillation.is_clip=2.0
        )
        ;;
    degrade)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.vcsd=True
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_mode=degrade
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_image_key=images_degraded
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$VCSD_FULL_LOGIT_ARG"
        )
        ;;
    qvis)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.vcsd=True
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_mode=qvis
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_image_key=images
            actor_rollout_ref.actor.self_distillation.vcsd_generic_prompt="Describe this image in detail."
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$VCSD_FULL_LOGIT_ARG"
        )
        ;;
    noimg)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.vcsd=True
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_mode=noimg
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$VCSD_FULL_LOGIT_ARG"
        )
        ;;
    black)
        EXPERIMENT_ARGS+=(
            actor_rollout_ref.actor.self_distillation.teacher_image_key=images
            actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=null
            actor_rollout_ref.actor.self_distillation.vcsd=True
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_mode=black
            actor_rollout_ref.actor.self_distillation.vcsd_ctrl_image_key=images
            actor_rollout_ref.actor.self_distillation.distillation_topk=null
            "$VCSD_FULL_LOGIT_ARG"
        )
        ;;
    *)
        echo "Unknown EXPERIMENT=$EXPERIMENT. Use visionopd|baseline|degrade|qvis|noimg|black." >&2
        exit 1
        ;;
esac

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-0}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

# --- Offline / no-tracking safety --------------------------------------------
# Force offline mode so no HF / dataset downloads are attempted over the network.
# The model must already be in the HF cache (MODEL_PATH resolves to the local
# snapshot above) and training data is a local parquet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Default HF cache to the home dir cache (where the complete snapshots live).
# Some environments point HF_HOME at a partial project cache that lacks config.json,
# which breaks AutoTokenizer/AutoConfig. Override with HF_HOME=... if you have a
# complete cache elsewhere.
if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi
# --- wandb (opt-in) -----------------------------------------------------------
# Optional: install wandb (`pip install wandb`) and set WANDB_API_KEY for online logging
# wandb uses WANDB_API_KEY from your environment.
# Enable with WANDB_ENABLE=1 (sets logger to console+tensorboard+wandb and online mode).
# Default: wandb disabled (console + tensorboard only).
if [[ "${WANDB_ENABLE:-0}" == "1" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
    unset WANDB_DISABLED 2>/dev/null || true
    if python3 - <<'PY' >/dev/null 2>&1
import tensorboard  # noqa: F401
PY
    then
        TRAINER_LOGGER='["console","tensorboard","wandb"]'
    else
        echo "WARNING: tensorboard is not installed; using console+wandb logger." >&2
        TRAINER_LOGGER='["console","wandb"]'
    fi
else
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
    if [[ -n "${TRAINER_LOGGER:-}" ]]; then
        TRAINER_LOGGER="$TRAINER_LOGGER"
    elif python3 - <<'PY' >/dev/null 2>&1
import tensorboard  # noqa: F401
PY
    then
        TRAINER_LOGGER='["console","tensorboard"]'
    else
        echo "WARNING: tensorboard is not installed; using console logger only." >&2
        TRAINER_LOGGER='["console"]'
    fi
fi

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running: $EXPERIMENT_NAME"
echo "Experiment: $EXPERIMENT"
echo "Model size: $MODEL_SIZE  (path: $MODEL_PATH)"
echo "GPUs: ${TRAINER_N_GPUS_PER_NODE}  (offline=$HF_HUB_OFFLINE, wandb=$WANDB_MODE)"
echo "Train file: $TASK_TRAIN_FILE"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.vcsd_temperature=2.0 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXPERIMENT_ARGS[@]}" \
    "${ANSWER_VAL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=vcsd_spar_vero_rl
#SBATCH --output=vcsd_spar_vero_rl_%j.out
#SBATCH --error=vcsd_spar_vero_rl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:2
#SBATCH --time=8:00:00
# Job 57684347 OOMed at the 192G cap during the first 96x8 multimodal rollout.
# Keep the worker-count safeguards and provide enough host RAM for the rollout
# payloads while retaining a conservative per-step batch size.
# Job 57796193 timed out after 4h at ~57% of the 8192-token prompt filter.
#SBATCH --mem=512G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leihan.chen@torontomu.ca

set -euo pipefail

# On Fir, Slurm copies the batch script under /localscratch/spool/slurmd and
# BASH_SOURCE points there — never use that to locate the repo. Prefer the
# directory from which sbatch was invoked (usually the VCSD checkout).
resolve_repo_root() {
    local candidate=""
    if [[ -n "${VCSD_ROOT:-}" ]]; then
        candidate="${VCSD_ROOT}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if [[ -f "${SLURM_SUBMIT_DIR}/scripts/run_experiment_contrast_standard_spar_vero_rl.sh" ]]; then
            candidate="${SLURM_SUBMIT_DIR}"
        elif [[ -f "${SLURM_SUBMIT_DIR}/run_experiment_contrast_standard_spar_vero_rl.sh" ]]; then
            candidate="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
        else
            candidate="${SLURM_SUBMIT_DIR}"
        fi
    else
        candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
    printf '%s' "${candidate}"
}

REPO_ROOT="$(resolve_repo_root)"
CONTAINER_ROOT="/opt/VCSD"
VCSD_IMAGE="${VCSD_IMAGE:-${REPO_ROOT}/VCSD.sif}"
HOST_TRAIN_SCRIPT="${REPO_ROOT}/scripts/run_experiment_contrast_standard_spar_vero_rl.sh"
CONTAINER_TRAIN_SCRIPT="${CONTAINER_ROOT}/scripts/run_experiment_contrast_standard_spar_vero_rl.sh"

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD=singularity
else
    echo "ERROR: neither apptainer nor singularity is available on PATH." >&2
    exit 1
fi
if [[ ! -x "$HOST_TRAIN_SCRIPT" ]]; then
    echo "ERROR: training launcher not found or not executable: $HOST_TRAIN_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$VCSD_IMAGE" ]]; then
    echo "ERROR: container image not found: $VCSD_IMAGE" >&2
    echo "       Build it first: $CONTAINER_CMD build --fakeroot $VCSD_IMAGE ${REPO_ROOT}/VCSD.def" >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/data/SPAR-7M" ]]; then
    echo "ERROR: SPAR image tree not found: ${REPO_ROOT}/data/SPAR-7M" >&2
    exit 1
fi

cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-vcsd}"
export WANDB_ENTITY="${WANDB_ENTITY:-cvis_tmu}"
export PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT}}"

export HF_HOME="${HF_HOME:-/project/def-wangcs/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
export DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/spar-vero-rl}"
export SPAR_IMAGE_ROOT="${SPAR_IMAGE_ROOT:-${REPO_ROOT}/data}"
export SPAR_VAL_ENABLE="${SPAR_VAL_ENABLE:-1}"
export SPAR_VAL_BATCH_SIZE="${SPAR_VAL_BATCH_SIZE:-16}"
export SPAR_VAL_TEST_FREQ="${SPAR_VAL_TEST_FREQ:-10}"
export SPAR_VAL_BEFORE_TRAIN="${SPAR_VAL_BEFORE_TRAIN:-True}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
# Do not map SLURM_CPUS_PER_TASK onto Python workers. 32 filter + 32 dataloader
# processes were the 57542414 host-RAM cliff. Keep dataloader workers at four;
# raise prompt-filter workers to 16 so the 8192-token VL filter can finish.
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-4}"
export SPAR_FILTER_WORKERS="${SPAR_FILTER_WORKERS:-16}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Ray 2.53.0 does not consume RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION;
# pass an explicit ray.init object_store_memory override below instead.
export RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$DATASET_DIR"

# Pass paths as seen inside the container. The repository bind includes the
# local Parquet files, SPAR-7M image tree, checkpoints, and rollout outputs.
pass_into_container() {
    local name="$1"
    local value="$2"
    export "APPTAINERENV_${name}=$value"
    export "SINGULARITYENV_${name}=$value"
}

if [[ -f /etc/ssl/certs/ca-bundle.crt ]]; then
    CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
elif [[ -f /etc/pki/tls/certs/ca-bundle.crt ]]; then
    CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
else
    CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

pass_into_container HF_HOME /hf_cache
pass_into_container HF_HUB_CACHE /hf_cache/hub
pass_into_container HF_DATASETS_CACHE /hf_cache/datasets
pass_into_container HF_HUB_OFFLINE "$HF_HUB_OFFLINE"
pass_into_container TRANSFORMERS_OFFLINE "$TRANSFORMERS_OFFLINE"
pass_into_container HF_DATASETS_OFFLINE "$HF_DATASETS_OFFLINE"
pass_into_container SSL_CERT_FILE "$CA_BUNDLE"
pass_into_container SSL_CERT_DIR /etc/ssl/certs
pass_into_container REQUESTS_CA_BUNDLE "$CA_BUNDLE"
pass_into_container CURL_CA_BUNDLE "$CA_BUNDLE"
pass_into_container MODEL_PATH "$MODEL_PATH"
pass_into_container DATASET_DIR "${CONTAINER_ROOT}/data/spar-vero-rl"
pass_into_container SPAR_IMAGE_ROOT "${CONTAINER_ROOT}/data"
pass_into_container SPAR_VAL_ENABLE "$SPAR_VAL_ENABLE"
pass_into_container SPAR_VAL_BATCH_SIZE "$SPAR_VAL_BATCH_SIZE"
pass_into_container SPAR_VAL_TEST_FREQ "$SPAR_VAL_TEST_FREQ"
pass_into_container SPAR_VAL_BEFORE_TRAIN "$SPAR_VAL_BEFORE_TRAIN"
pass_into_container TRAINER_N_GPUS_PER_NODE "$TRAINER_N_GPUS_PER_NODE"
pass_into_container DATA_DATALOADER_NUM_WORKERS "$DATA_DATALOADER_NUM_WORKERS"
pass_into_container SPAR_FILTER_WORKERS "$SPAR_FILTER_WORKERS"
pass_into_container ROLLOUT_AGENT_NUM_WORKERS "$ROLLOUT_AGENT_NUM_WORKERS"
pass_into_container TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE"
pass_into_container PPO_MIMI_BATCH_SIZE "$PPO_MIMI_BATCH_SIZE"
pass_into_container ROLLOUT_N "$ROLLOUT_N"
pass_into_container MAX_PROMPT_LENGTH "$MAX_PROMPT_LENGTH"
pass_into_container MAX_RESPONSE_LENGTH "$MAX_RESPONSE_LENGTH"
pass_into_container OMP_NUM_THREADS "$OMP_NUM_THREADS"
pass_into_container MKL_NUM_THREADS "$MKL_NUM_THREADS"
pass_into_container OPENBLAS_NUM_THREADS "$OPENBLAS_NUM_THREADS"
pass_into_container TOKENIZERS_PARALLELISM "$TOKENIZERS_PARALLELISM"
pass_into_container RAY_OBJECT_STORE_MEMORY_BYTES "$RAY_OBJECT_STORE_MEMORY_BYTES"
pass_into_container PREPEND_PATH /opt/vcsd/bin
pass_into_container WANDB_ENABLE "$WANDB_ENABLE"
pass_into_container WANDB_MODE "$WANDB_MODE"
pass_into_container WANDB_PROJECT "$WANDB_PROJECT"
pass_into_container WANDB_ENTITY "$WANDB_ENTITY"
pass_into_container PROJECT_NAME "$PROJECT_NAME"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    pass_into_container WANDB_API_KEY "$WANDB_API_KEY"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    pass_into_container HF_TOKEN "$HF_TOKEN"
    pass_into_container HUGGING_FACE_HUB_TOKEN "$HF_TOKEN"
fi

echo "Repository: $REPO_ROOT"
echo "Container runtime: $CONTAINER_CMD"
echo "Container image: $VCSD_IMAGE"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "GPUs: $TRAINER_N_GPUS_PER_NODE"
echo "Model: $MODEL_PATH (HF offline=$HF_HUB_OFFLINE, datasets offline=$HF_DATASETS_OFFLINE, Transformers offline=$TRANSFORMERS_OFFLINE)"
echo "TLS CA bundle: $CA_BUNDLE"
echo "Dataset cache: $DATASET_DIR"
echo "SPAR image root: $SPAR_IMAGE_ROOT"
echo "SPAR validation: $SPAR_VAL_ENABLE (every $SPAR_VAL_TEST_FREQ steps)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}  mem=${SLURM_MEM_PER_NODE:-512G}  dataloader workers: $DATA_DATALOADER_NUM_WORKERS  filter workers: $SPAR_FILTER_WORKERS  agent workers: $ROLLOUT_AGENT_NUM_WORKERS"
echo "Batch: train=$TRAIN_BATCH_SIZE  ppo_mini=$PPO_MIMI_BATCH_SIZE  rollout_n=$ROLLOUT_N"
echo "Threads: OMP=$OMP_NUM_THREADS  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM  ray object store bytes=$RAY_OBJECT_STORE_MEMORY_BYTES"
echo "Max tokens: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH"
echo "W&B: enable=$WANDB_ENABLE mode=$WANDB_MODE project=$WANDB_PROJECT entity=$WANDB_ENTITY"

start_time=$(date +%s)

srun "$CONTAINER_CMD" exec --nv \
    --bind "${REPO_ROOT}:${CONTAINER_ROOT}" \
    --bind "${HF_HOME}:/hf_cache" \
    --bind /etc/ssl/certs:/etc/ssl/certs:ro \
    --bind /etc/pki:/etc/pki:ro \
    --pwd "$CONTAINER_ROOT" \
    "$VCSD_IMAGE" \
    bash "$CONTAINER_TRAIN_SCRIPT" \
    "+ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY_BYTES}" \
    "$@"

end_time=$(date +%s)
duration=$((end_time - start_time))
printf 'Training time: %dh %dm %ds\n' \
    "$((duration / 3600))" \
    "$(((duration % 3600) / 60))" \
    "$((duration % 60))"

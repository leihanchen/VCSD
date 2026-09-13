#!/bin/bash
#SBATCH --account=def-wangcs
#SBATCH --job-name=vcsd_spar_vero_rl_filter
#SBATCH --output=vcsd_spar_vero_rl_filter_%j.out
#SBATCH --error=vcsd_spar_vero_rl_filter_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
# CPU-only precompute of SPAR missing-image + 8192-token VL length filters.
# Job 57796193 spent a 4h GPU allocation on this step and timed out at ~58%.
#SBATCH --mem=256G
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
        if [[ -f "${SLURM_SUBMIT_DIR}/scripts/precompute_spar_vero_rl_filtered.py" ]]; then
            candidate="${SLURM_SUBMIT_DIR}"
        elif [[ -f "${SLURM_SUBMIT_DIR}/precompute_spar_vero_rl_filtered.py" ]]; then
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
HOST_SCRIPT="${REPO_ROOT}/scripts/precompute_spar_vero_rl_filtered.py"
CONTAINER_SCRIPT="${CONTAINER_ROOT}/scripts/precompute_spar_vero_rl_filtered.py"

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
if [[ ! -f "$HOST_SCRIPT" ]]; then
    echo "ERROR: precompute script not found: $HOST_SCRIPT" >&2
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

export HF_HOME="${HF_HOME:-/project/def-wangcs/${USER}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
export SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-cvis-tmu/spar-vero-rl}"
export HF_DATASET_ID="${HF_DATASET_ID:-cvis-tmu/spar-vero-rl-filtered}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export SPAR_FILTER_WORKERS="${SPAR_FILTER_WORKERS:-16}"
export FILTER_SPLITS="${FILTER_SPLITS:-train,test}"
export HF_PRIVATE="${HF_PRIVATE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

pass_into_container() {
    local name="$1"
    local value="$2"
    export "APPTAINERENV_${name}=$value"
    export "SINGULARITYENV_${name}=$value"
}

pass_into_container HF_HOME /hf_cache
pass_into_container HF_HUB_CACHE /hf_cache/hub
pass_into_container HF_DATASETS_CACHE /hf_cache/datasets
pass_into_container HF_HUB_OFFLINE "$HF_HUB_OFFLINE"
pass_into_container HF_DATASETS_OFFLINE "$HF_DATASETS_OFFLINE"
pass_into_container MODEL_PATH "$MODEL_PATH"
pass_into_container SOURCE_DATASET_ID "$SOURCE_DATASET_ID"
pass_into_container HF_DATASET_ID "$HF_DATASET_ID"
pass_into_container MAX_PROMPT_LENGTH "$MAX_PROMPT_LENGTH"
pass_into_container SPAR_FILTER_WORKERS "$SPAR_FILTER_WORKERS"
pass_into_container OMP_NUM_THREADS "$OMP_NUM_THREADS"
pass_into_container MKL_NUM_THREADS "$MKL_NUM_THREADS"
pass_into_container TOKENIZERS_PARALLELISM "$TOKENIZERS_PARALLELISM"
pass_into_container PREPEND_PATH /opt/vcsd/bin
if [[ -n "${HF_TOKEN:-}" ]]; then
    pass_into_container HF_TOKEN "$HF_TOKEN"
    pass_into_container HUGGING_FACE_HUB_TOKEN "$HF_TOKEN"
fi
# Slurm inherits the host SSL_CERT_FILE (/etc/pki/tls/certs/ca-bundle.crt on
# AlmaLinux). That path is absent in the container, and bind-mounting
# /etc/ssl/certs overlays broken host symlinks. Bind the resolved PEM instead.
HOST_CA_BUNDLE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}"
if [[ -e "$HOST_CA_BUNDLE" ]]; then
    HOST_CA_BUNDLE="$(readlink -f "$HOST_CA_BUNDLE")"
fi
CONTAINER_CA_BUNDLE="/certs/ca-bundle.crt"
pass_into_container SSL_CERT_FILE "$CONTAINER_CA_BUNDLE"
pass_into_container REQUESTS_CA_BUNDLE "$CONTAINER_CA_BUNDLE"
pass_into_container CURL_CA_BUNDLE "$CONTAINER_CA_BUNDLE"

UPLOAD_ARGS=(--upload)
if [[ "$HF_PRIVATE" == "1" ]]; then
    UPLOAD_ARGS+=(--private)
fi

echo "Repo: $REPO_ROOT"
echo "Container: $CONTAINER_CMD $VCSD_IMAGE"
echo "Node: ${SLURMD_NODENAME:-unknown}  CPUs: ${SLURM_CPUS_PER_TASK:-unknown}  mem=${SLURM_MEM_PER_NODE:-256G}"
echo "Model: $MODEL_PATH  max_prompt_length=$MAX_PROMPT_LENGTH  filter workers=$SPAR_FILTER_WORKERS"
echo "Source: $SOURCE_DATASET_ID -> ${CONTAINER_ROOT}/data/spar-vero-rl"
echo "Output: ${CONTAINER_ROOT}/data/spar-vero-rl-filtered  HF repo=$HF_DATASET_ID private=$HF_PRIVATE"

START_TS=$(date +%s)
APPTAINER_BINDS=(
    --bind "${REPO_ROOT}:${CONTAINER_ROOT}"
    --bind "${HF_HOME}:/hf_cache"
)
if [[ -n "${HOST_CA_BUNDLE:-}" && -f "$HOST_CA_BUNDLE" ]]; then
    APPTAINER_BINDS+=(--bind "${HOST_CA_BUNDLE}:${CONTAINER_CA_BUNDLE}:ro")
    echo "TLS CA: $HOST_CA_BUNDLE -> $CONTAINER_CA_BUNDLE"
else
    echo "WARNING: host CA bundle not found; container TLS may fail" >&2
fi

srun "$CONTAINER_CMD" exec \
    "${APPTAINER_BINDS[@]}" \
    --pwd "$CONTAINER_ROOT" \
    "$VCSD_IMAGE" \
    python3 "$CONTAINER_SCRIPT" \
        --source-dir "${CONTAINER_ROOT}/data/spar-vero-rl" \
        --source-repo "$SOURCE_DATASET_ID" \
        --output-dir "${CONTAINER_ROOT}/data/spar-vero-rl-filtered" \
        --image-root "${CONTAINER_ROOT}/data" \
        --model-path "$MODEL_PATH" \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --num-workers "$SPAR_FILTER_WORKERS" \
        --splits "$FILTER_SPLITS" \
        --hf-repo "$HF_DATASET_ID" \
        "${UPLOAD_ARGS[@]}"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "Precompute finished in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"

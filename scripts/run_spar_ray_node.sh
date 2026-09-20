#!/bin/bash
# Run once per Slurm node inside the container; only rank zero runs the trainer.
set -euo pipefail
: "${RAY_HEAD_IP:?}" "${RAY_PORT:?}" "${RAY_JOB_STATE_DIR:?}"
: "${WORLD_SIZE:?}" "${TRAINER_N_GPUS_PER_NODE:?}"
rank="${SLURM_PROCID:-0}"
ray_pid=""
cleanup() {
    local code=$?
    if [[ "$rank" == 0 ]]; then
        printf '%s\n' "$code" > "${RAY_JOB_STATE_DIR}/finished"
    fi
    if [[ -n "$ray_pid" ]]; then
        kill "$ray_pid" 2>/dev/null || true
        wait "$ray_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
common=(--num-cpus="${SLURM_CPUS_PER_TASK:-32}"
        --num-gpus="$TRAINER_N_GPUS_PER_NODE"
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"
        --disable-usage-stats --block)
if [[ "$rank" == 0 ]]; then
    ray start --head --node-ip-address="$RAY_HEAD_IP" --port="$RAY_PORT" \
        --include-dashboard=false "${common[@]}" &
    ray_pid=$!
    export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
    python3 - <<'PY'
import os
import time
import ray

deadline = time.monotonic() + 300
expected_nodes = int(os.environ["WORLD_SIZE"])
expected_gpus = expected_nodes * int(os.environ["TRAINER_N_GPUS_PER_NODE"])
while time.monotonic() < deadline:
    try:
        if not ray.is_initialized():
            ray.init(address=os.environ["RAY_ADDRESS"])
        live = [n for n in ray.nodes() if n["Alive"]]
        if len(live) == expected_nodes and ray.cluster_resources().get("GPU", 0) >= expected_gpus:
            print(f"Ray ready: {len(live)} nodes, {expected_gpus} GPUs", flush=True)
            ray.shutdown()
            break
    except (ConnectionError, RuntimeError):
        pass
    time.sleep(2)
else:
    raise RuntimeError(f"Ray did not register {expected_nodes} nodes/{expected_gpus} GPUs within 300s")
PY
    bash "$@" "+ray_kwargs.ray_init.address=${RAY_ADDRESS}"
else
    # Wait for the head's GCS port before starting the worker.
    python3 - <<'PY'
import os
import socket
import time

deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        with socket.create_connection((os.environ["RAY_HEAD_IP"], int(os.environ["RAY_PORT"])), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError("Ray head did not become reachable within 300s")
PY
    ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" "${common[@]}" &
    ray_pid=$!
    while [[ ! -f "${RAY_JOB_STATE_DIR}/finished" ]]; do
        if ! kill -0 "$ray_pid" 2>/dev/null; then
            wait "$ray_pid" || true
            echo "Ray worker exited before training completed" >&2
            exit 1
        fi
        sleep 2
    done
    exit "$(cat "${RAY_JOB_STATE_DIR}/finished")"
fi

import os
import subprocess
from pathlib import Path


def _run_head(tmp_path, exit_code):
    root = Path(__file__).resolve().parents[1]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    trace = tmp_path / "trace"
    scripts = {
        "ray": '#!/bin/bash\necho "ray $*" >> "$TRACE"\nexec sleep 60\n',
        "python3": '#!/bin/bash\ncat >/dev/null\nexit 0\n',
        "train": f'#!/bin/bash\necho "train $*" >> "$TRACE"\nexit {exit_code}\n',
    }
    for name, body in scripts.items():
        file = bindir / name
        file.write_text(body)
        file.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", TRACE=str(trace),
               SLURM_PROCID="0", SLURM_CPUS_PER_TASK="4", TRAINER_N_GPUS_PER_NODE="4",
               WORLD_SIZE="2", RAY_HEAD_IP="127.0.0.1", RAY_PORT="16379",
               RAY_JOB_STATE_DIR=str(tmp_path), RAY_OBJECT_STORE_MEMORY_BYTES="100000000")
    result = subprocess.run(["bash", str(root / "scripts/run_spar_ray_node.sh"),
                             str(bindir / "train"), "sentinel=1"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    text = trace.read_text()
    assert "--head" in text and "--num-gpus=4" in text
    assert text.count("train ") == 1
    assert "sentinel=1" in text
    assert "ray_kwargs.ray_init.address" in text
    assert (tmp_path / "finished").read_text().strip() == str(exit_code)


def test_head_runs_driver_once_and_stops_ray(tmp_path):
    _run_head(tmp_path, 0)


def test_head_propagates_driver_failure_to_workers(tmp_path):
    _run_head(tmp_path, 7)


def _run_worker(tmp_path, ray_body, finished_code=None):
    root = Path(__file__).resolve().parents[1]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    trace = tmp_path / "trace"
    for name, body in {
        "ray": '#!/bin/bash\necho "ray $*" >> "$TRACE"\n' + ray_body,
        "python3": '#!/bin/bash\ncat >/dev/null\nexit 0\n',
    }.items():
        executable = bindir / name
        executable.write_text(body)
        executable.chmod(0o755)
    # The fake Ray process publishes the driver's status after registering.
    if finished_code is not None:
        ray = bindir / "ray"
        ray.write_text(ray.read_text().replace(
            "exec sleep 60", f'echo {finished_code} > "$RAY_JOB_STATE_DIR/finished"\nexec sleep 60'))
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", TRACE=str(trace),
               SLURM_PROCID="1", TRAINER_N_GPUS_PER_NODE="4", WORLD_SIZE="2",
               RAY_HEAD_IP="127.0.0.1", RAY_PORT="16379", RAY_JOB_STATE_DIR=str(tmp_path))
    result = subprocess.run(["bash", str(root / "scripts/run_spar_ray_node.sh"),
                             "/must-not-run-trainer"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert "--address=127.0.0.1:16379" in trace.read_text()
    assert "--head" not in trace.read_text()
    return result


def test_worker_returns_driver_status_without_running_trainer(tmp_path):
    result = _run_worker(tmp_path, "exec sleep 60\n", finished_code=7)
    assert result.returncode == 7, result.stderr


def test_worker_failure_is_not_reported_as_success(tmp_path):
    result = _run_worker(tmp_path, "exit 3\n")
    assert result.returncode == 1
    assert "Ray worker exited before training completed" in result.stderr

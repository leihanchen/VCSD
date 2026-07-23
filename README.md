# VCSD — Visual Contrast Self-Distillation for VLMs

Training code for **contrast-sharpened self-distillation** ("ours"): an RL fine-tuning method for
vision-language models that sharpens the model's own visual grounding by distilling toward a
**contrast target** built from the gap between a normal-image teacher pass and a control
(e.g. blacked-out image) pass.

This code is derived from [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD), which is
itself built on the [verl](https://github.com/volcengine/verl) GRPO/PPO trainer (the `verl/` package
here). The VCSD method-specific code lives in a few files; everything else is the inherited RL
framework. See [Acknowledgments](#acknowledgments).

## Method in one paragraph

For each token, we form a contrast-sharpened distillation target

```
target = softmax( log p_hi + α · (log p_hi − log p_ctrl) )
```

where `p_hi` is the teacher distribution on the real image and `p_ctrl` is the teacher distribution
on a control input (image removed / blacked-out / degraded). A **β-plausibility mask** restricts the
support to `{w : p_hi(w) ≥ β · max p_hi}`, so mass is only redistributed among tokens the plain
teacher already considers plausible. The student is trained toward this target with a plain
**token-averaged full-vocab KL** over the response (no per-token reweighting, no gating), alongside
the RL objective.

## Key files

| File | Role |
|---|---|
| `verl/trainer/ppo/vcsd.py` | `build_contrast_target` + token-averaged `vcsd_kd_loss` |
| `verl/workers/actor/dp_actor.py` | teacher hi/ctrl forward passes + distillation loss call |
| `verl/workers/config/actor.py` · `verl/trainer/config/actor/actor.yaml` | `vcsd_*` config knobs |
| `verl/trainer/config/vopd.yaml` | training config (hydra) |
| `scripts/run_vcsd.sh` | training entry point (env-driven, hydra overrides) |
| `scripts/run_experiment_contrast_standard.sh` | launcher: `α=1.0`, `β=0.1`, black-image control |

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

Requires a CUDA GPU stack compatible with the pinned `torch` / `vllm` / `flash-attn` in
`requirements.txt` (see the notes there for building flash-attn / causal-conv1d from source).

## Data

Training expects Parquet files under a `DATA_DIR` (default `data/`):

- `train.parquet` — training prompts with an image column
- `train_answer.parquet` / `val_answer.parquet` — answer-conditioned val splits

Prepare helpers are in `scripts/` (`prepare_answer_val_split.py`,
`prepare_degraded_images.py`). Datasets are **not** included in this repo — point `TASK_TRAIN_FILE`
/ `ANSWER_VAL_TRAIN_FILE` at your own Parquet files.

## Train ("ours")

```bash
MODEL_SIZE=2B \
MODEL_PATH=<path-or-HF-name, e.g. Qwen/Qwen3-VL-2B-Instruct> \
EXPERIMENT_NAME=vcsd-qwen3vl-2b \
TRAINER_N_GPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
ANSWER_VAL_TRAIN_FILE=data/train.parquet \
bash scripts/run_experiment_contrast_standard.sh \
  data.filter_overlong_prompts=True trainer.total_training_steps=150
```

Key method knobs (hydra overrides, defaults set by the launcher):

- `actor_rollout_ref.actor.self_distillation.vcsd_target_mode=contrast`
- `...vcsd_contrast_alpha=1.0` — tilt strength α
- `...vcsd_contrast_beta=0.1` — plausibility mask β (set `0.0` to disable → training collapses)
- `...vcsd_ctrl_mode=black` — control input (`black` / `degrade` / `noimg`)
- `...vcsd_contrast_exclude_token_ids=[151643,151645]` — vocab ids exempt from the tilt (EOS/im_end)

Checkpoints land in `checkpoints/<EXPERIMENT_NAME>/global_step_*`; merge FSDP shards with
`python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <ckpt>`.

## Acknowledgments

This repository is based on [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD) and the
[verl](https://github.com/volcengine/verl) reinforcement-learning framework it builds on. The `verl/`
package here is inherited from those projects; VCSD adds the contrast-sharpened self-distillation
target and loss (`verl/trainer/ppo/vcsd.py`, the `vcsd_*` config knobs, and the training launchers).
We thank the authors of both projects.

## License

Apache-2.0 — see `LICENSE`. Inherited `verl/` code retains its original copyright headers.

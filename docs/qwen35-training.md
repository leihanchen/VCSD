# Qwen3.5 SPAR training

`scripts/run_spar_vero_rl.slurm` defaults to Qwen/Qwen3.5-2B, two Fir nodes
with four H100 GPUs each, and a 48-hour limit. The global prompt batch is 8,
with four rollouts per prompt (32 generated sequences per training step).
Validation runs before training and every 50 steps. Checkpoints remain every
10 steps. The model-specific experiment name gives this run a separate
checkpoint directory from the previous Qwen3-VL run.

The launcher starts a Ray head and worker in their respective containers,
waits for both nodes and eight GPUs, then starts exactly one training driver.
The worker propagates the driver's completion status; early worker failure
fails the Slurm step.

## Container

The original `VCSD.sif` has Transformers 4.57.6, which does not recognize
`qwen3_5`. The launcher now uses `VCSD-qwen35.sif`. Rebuild from the repository
root with the original image available:

```bash
apptainer build --fakeroot VCSD-qwen35.sif VCSD-qwen35.def
```

This pins Transformers to revision
`cc7ab9be508ce6ed3637bba9e50367b29b742dc6` (5.3.0.dev0), retaining vLLM 0.18
and the original CUDA/PyTorch stack. This combination follows the
[upstream Qwen3.5 FSDP recipe](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh).
It conflicts with vLLM 0.18's declared Transformers `<5` package constraint;
successful imports alone do not prove full GPU runtime compatibility.

Qwen3.5 uses FSDP2, padded execution, and one sequence per actor microbatch.
The launcher reads image patch size and pad/EOS IDs from the selected model
instead of using Qwen3-VL constants. It rechecks prompt lengths because the
published SPAR subset was filtered using the older processor.

## Verification and limits

Verified: image build and architecture imports, vLLM architecture registration,
actual model tokenizer/processor loading, a tiny hybrid-attention model CPU
forward/backward pass, resolved training configuration, four mocked Ray
launcher lifecycle tests, shell syntax, and Slurm `--test-only` acceptance.

Full model GPU rollout, distributed FSDP2 training, and throughput have not
been validated. No training job was submitted. The 48-hour limit does not
guarantee a full epoch; re-filtering, the architecture change, and cross-node
communication require fresh timing measurements.

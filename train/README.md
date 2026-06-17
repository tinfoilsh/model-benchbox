# Training CC Benchmarks

This harness runs NeMo AutoModel post-training workloads inside the benchbox CVM
and writes results to `/dev/shm/results/train`. The image installs
`nemo-automodel[cli,vlm]` on top of the repo's standard vLLM base image because
the full NVIDIA NeMo AutoModel container is too large for the GitHub Actions build
runner.

Run the same release twice on `control.inf8.tinfoil.sh`:

```bash
tinfoil container create benchtrain-gemma-h200-cc-on \
  --repo tinfoilsh/model-benchbox --tag <tag> --debug \
  --host control.inf8.tinfoil.sh --ssh-key tanya-key \
  --secret GITHUB_TOKEN --secret HF_TOKEN --secret TINFOIL_API_KEY

tinfoil container create benchtrain-gemma-h200-cc-off \
  --repo tinfoilsh/model-benchbox --tag <tag> --debug \
  --host control.inf8.tinfoil.sh --ssh-key tanya-key \
  --secret GITHUB_TOKEN --secret HF_TOKEN --secret TINFOIL_API_KEY \
  --disable-cc-mode --yes
```

Inside the `benchbox` container:

```bash
cd /workspace
train/run_gemma_training.sh cc-on smoke-e4b-peft run1
train/run_gemma_training.sh cc-on gemma31b-peft run1
train/run_gemma_training.sh cc-on gemma31b-sft run1
```

Scenarios:

- `smoke-e4b-peft`: `google/gemma-4-E4B-it`, LoRA, default 50 steps.
- `gemma31b-peft`: `google/gemma-4-31B-it`, LoRA, default 200 steps.
- `gemma31b-sft`: `google/gemma-4-31B-it`, full SFT with FSDP2 activation checkpointing,
  default 100 steps.

The 31B scenarios prefer the mounted MPK path
`/tinfoil/mpk/mpk-d2f38032f5faaa8a45b433157d0f25da1e533cf233784c7438fe9f46d1fbc3f4`
and fall back to Hugging Face if it is absent.

Useful overrides:

```bash
SMOKE_STEPS=20 GEMMA31B_PEFT_STEPS=100 GEMMA31B_SFT_STEPS=100 \
  train/run_training_suite.sh cc-on run1

PIN_MEMORY=false train/run_gemma_training.sh cc-on smoke-e4b-peft pin-off
```

Each scenario directory contains:

- `provenance.txt`
- `cc-mode.txt`
- `gpu-telemetry.csv`
- `train.log`
- `summary.json`
- `manifest.json`

## Kimi K2.6 VLM LoRA

The Kimi runner targets the inf14-staged `nvidia/Kimi-K2.6-NVFP4` MPK at:

```text
/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8
```

Run one smoke step sequence before committing to the longer arm:

```bash
cd /workspace
KIMI26_SMOKE_STEPS=5 WARMUP_STEPS=1 train/run_kimi_training.sh cc-on kimi26-vl-lora-smoke run1
KIMI26_LORA_STEPS=120 WARMUP_STEPS=20 train/run_kimi_training.sh cc-on kimi26-vl-lora run1
```

The Kimi script refuses to fall back to a Hugging Face download unless
`ALLOW_HF_KIMI_DOWNLOAD=true` is set, because the checkpoint is too large for an
accidental network pull. Useful tuning overrides:

```bash
KIMI26_MAX_LENGTH=512 KIMI26_LORA_STEPS=20 \
  train/run_kimi_training.sh cc-on kimi26-vl-lora-smoke debug

KIMI26_DISPATCHER=torch train/run_kimi_training.sh cc-on kimi26-vl-lora-smoke torch-dispatch
```

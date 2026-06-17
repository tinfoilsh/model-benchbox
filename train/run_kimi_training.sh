#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: train/run_kimi_training.sh <cc-on|cc-off> <kimi26-vl-lora-smoke|kimi26-vl-lora> [label]

Environment overrides:
  NPROC_PER_NODE          torchrun worker count (default: 8)
  KIMI26_SMOKE_STEPS     default 5
  KIMI26_LORA_STEPS      default 120
  KIMI26_MODEL           default: local inf14 MPK mount
  ALLOW_HF_KIMI_DOWNLOAD set true to allow fallback to nvidia/Kimi-K2.6-NVFP4
  KIMI26_GLOBAL_BATCH    default 8
  KIMI26_LOCAL_BATCH     default 1
  KIMI26_MAX_LENGTH      default 1024
  KIMI26_EXPERTS         default torch_mm
  KIMI26_DISPATCHER      default deepep
  PIN_MEMORY             dataloader pin_memory override, true/false (default: true)
  WARMUP_STEPS           step records to discard in summary (default: 10)
EOF
}

COND=${1:-}
SCENARIO=${2:-}
LABEL=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
[ -n "$COND" ] && [ -n "$SCENARIO" ] || { usage; exit 2; }
case "$COND" in cc-on|cc-off) ;; *) usage; exit 2 ;; esac

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$TRAIN_ROOT/configs"
SCRIPTS_DIR="$TRAIN_ROOT/scripts"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PIN_MEMORY=${PIN_MEMORY:-true}
WARMUP_STEPS=${WARMUP_STEPS:-10}
KIMI26_GLOBAL_BATCH=${KIMI26_GLOBAL_BATCH:-8}
KIMI26_LOCAL_BATCH=${KIMI26_LOCAL_BATCH:-1}
KIMI26_MAX_LENGTH=${KIMI26_MAX_LENGTH:-1024}
KIMI26_EXPERTS=${KIMI26_EXPERTS:-torch_mm}
KIMI26_DISPATCHER=${KIMI26_DISPATCHER:-deepep}

LOCAL_KIMI="/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8"
if [ -n "${KIMI26_MODEL:-}" ]; then
  MODEL="$KIMI26_MODEL"
elif [ -e "$LOCAL_KIMI" ]; then
  MODEL="$LOCAL_KIMI"
elif [ "${ALLOW_HF_KIMI_DOWNLOAD:-false}" = "true" ]; then
  MODEL="nvidia/Kimi-K2.6-NVFP4"
else
  echo "[train] missing local Kimi MPK at $LOCAL_KIMI" >&2
  echo "[train] set KIMI26_MODEL or ALLOW_HF_KIMI_DOWNLOAD=true to override" >&2
  exit 1
fi

case "$SCENARIO" in
  kimi26-vl-lora-smoke)
    CONFIG="$CONFIG_DIR/kimi_k2_6_vl_lora.yaml"
    STEPS=${KIMI26_SMOKE_STEPS:-5}
    ;;
  kimi26-vl-lora)
    CONFIG="$CONFIG_DIR/kimi_k2_6_vl_lora.yaml"
    STEPS=${KIMI26_LORA_STEPS:-120}
    ;;
  *)
    usage; exit 2
    ;;
esac

export TMPDIR=${TMPDIR:-/dev/shm/tmp}
export HF_HOME=${HF_HOME:-/dev/shm/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/dev/shm/hf/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/triton}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

OUT=${OUT:-/dev/shm/results/train/${COND}/${LABEL}/${SCENARIO}}
mkdir -p "$OUT"

AUTOMODEL_DIR=${AUTOMODEL_DIR:-/opt/Automodel}
if [ ! -f "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" ]; then
  AUTOMODEL_DIR=/dev/shm/Automodel
  if [ ! -f "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" ]; then
    git clone --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_DIR"
  fi
fi

cp "$CONFIG" "$OUT/config.yaml"
"$SCRIPTS_DIR/provenance.sh" > "$OUT/provenance.txt" 2>&1 || true

CC_MGM="$(nvidia-smi conf-compute -mgm 2>&1 || true)"
CC_F="$(nvidia-smi conf-compute -f 2>&1 || true)"
printf '%s\n\n%s\n' "$CC_MGM" "$CC_F" > "$OUT/cc-mode.txt"

cat > "$OUT/manifest.json" <<EOF
{
  "condition": "$COND",
  "scenario": "$SCENARIO",
  "label": "$LABEL",
  "model": "$MODEL",
  "config": "$CONFIG",
  "steps": $STEPS,
  "nproc_per_node": $NPROC_PER_NODE,
  "global_batch_size": $KIMI26_GLOBAL_BATCH,
  "local_batch_size": $KIMI26_LOCAL_BATCH,
  "max_length": $KIMI26_MAX_LENGTH,
  "experts": "$KIMI26_EXPERTS",
  "dispatcher": "$KIMI26_DISPATCHER",
  "pin_memory": "$PIN_MEMORY",
  "automodel_dir": "$AUTOMODEL_DIR",
  "output_dir": "$OUT",
  "cc_mgm": $(CC_MGM="$CC_MGM" python3 -c 'import json,os; print(json.dumps(os.environ.get("CC_MGM", "")))')
}
EOF

echo "[train] condition=$COND scenario=$SCENARIO label=$LABEL"
echo "[train] model=$MODEL"
echo "[train] steps=$STEPS nproc=$NPROC_PER_NODE global_batch=$KIMI26_GLOBAL_BATCH local_batch=$KIMI26_LOCAL_BATCH max_length=$KIMI26_MAX_LENGTH out=$OUT"
echo "[train] backend experts=$KIMI26_EXPERTS dispatcher=$KIMI26_DISPATCHER"

"$SCRIPTS_DIR/collect_gpu_telemetry.sh" "$OUT/gpu-telemetry.csv" "${TELEMETRY_INTERVAL:-2}" &
TELEM_PID=$!
cleanup() {
  kill "$TELEM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

START_EPOCH=$(date +%s)
set +e
torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
  "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" \
  -c "$CONFIG" \
  --model.pretrained_model_name_or_path "$MODEL" \
  --processor.pretrained_model_name_or_path "$MODEL" \
  --model.backend.experts "$KIMI26_EXPERTS" \
  --model.backend.dispatcher "$KIMI26_DISPATCHER" \
  --step_scheduler.max_steps "$STEPS" \
  --step_scheduler.global_batch_size "$KIMI26_GLOBAL_BATCH" \
  --step_scheduler.local_batch_size "$KIMI26_LOCAL_BATCH" \
  --step_scheduler.num_epochs 1 \
  --step_scheduler.ckpt_every_steps 1000000 \
  --step_scheduler.val_every_steps 1000000 \
  --checkpoint.enabled false \
  --checkpoint.checkpoint_dir "/dev/shm/checkpoints/${COND}/${LABEL}/${SCENARIO}" \
  --dataloader.pin_memory "$PIN_MEMORY" \
  --dataloader.collate_fn.max_length "$KIMI26_MAX_LENGTH" \
  --validation_dataloader.collate_fn.max_length "$KIMI26_MAX_LENGTH" \
  2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}
set -e
END_EPOCH=$(date +%s)
cleanup
trap - EXIT

python3 "$SCRIPTS_DIR/summarize_log.py" "$OUT/train.log" "$OUT/summary.json" "$WARMUP_STEPS" > "$OUT/summary.stdout.txt" 2>&1 || true

python3 - "$OUT/manifest.json" "$RC" "$START_EPOCH" "$END_EPOCH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["returncode"] = int(sys.argv[2])
data["start_epoch"] = int(sys.argv[3])
data["end_epoch"] = int(sys.argv[4])
data["wall_seconds"] = data["end_epoch"] - data["start_epoch"]
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "[train] returncode=$RC wall=$((END_EPOCH - START_EPOCH))s -> $OUT"
exit "$RC"

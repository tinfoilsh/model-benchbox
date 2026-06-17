#!/usr/bin/env bash
set -uo pipefail

sec() { printf '\n===== %s =====\n' "$*"; }

sec date; date -u
sec nvidia-smi; nvidia-smi 2>&1
sec "driver version"; cat /proc/driver/nvidia/version 2>&1
sec "conf-compute -f"; nvidia-smi conf-compute -f 2>&1
sec "conf-compute -e"; nvidia-smi conf-compute -e 2>&1
sec "conf-compute -d"; nvidia-smi conf-compute -d 2>&1
sec "conf-compute -grs"; nvidia-smi conf-compute -grs 2>&1
sec "conf-compute -mgm"; nvidia-smi conf-compute -mgm 2>&1
sec "topo -m"; nvidia-smi topo -m 2>&1
sec "gpu query"
nvidia-smi --query-gpu=index,name,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.total,memory.used --format=csv 2>&1
sec "cpu count + numa"
nproc 2>&1
lscpu 2>&1 | head -40
numactl --hardware 2>&1 || true
sec "kernel cmdline"; cat /proc/cmdline 2>&1
sec "tdx/swiotlb dmesg"
dmesg 2>/dev/null | grep -iE 'tdx|swiotlb|conf.*comput' | head -80 || true
sec "python packages"
python3 - <<'PY' 2>&1
mods = ["torch", "transformers", "nemo_automodel", "datasets", "torchdata"]
for mod in mods:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "unknown"))
    except Exception as exc:
        print(mod, "ERROR", repr(exc))
try:
    import torch
    print("cuda", torch.version.cuda)
    print("nccl", torch.cuda.nccl.version())
    print("gpus", torch.cuda.device_count())
except Exception as exc:
    print("torch cuda probe error", repr(exc))
PY
sec "pip freeze selected"
pip freeze 2>/dev/null | grep -iE '^(nemo|torch|transformers|datasets|accelerate|peft|liger|flash|triton|numpy|pandas|huggingface)' || true

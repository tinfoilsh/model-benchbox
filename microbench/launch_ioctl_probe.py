#!/usr/bin/env python3
"""Launch-only probe for syscall scaling checks."""
import json
import os
import time

import torch


N = int(os.environ.get("LAUNCH_IOCTL_N", "1000"))
WARMUP_N = int(os.environ.get("LAUNCH_IOCTL_WARMUP_N", "20"))

x = torch.ones(256, device="cuda:0")
for _ in range(WARMUP_N):
    x.add_(1.0)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    x.add_(1.0)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

print(
    json.dumps(
        {
            "device": torch.cuda.get_device_name(0),
            "launches": N,
            "elapsed_s": dt,
            "us_per_launch": (dt / N * 1e6) if N else None,
        },
        indent=1,
    )
)

#!/usr/bin/env python3
"""GPU allocatable-memory probe (CPR carve-out measurement): alloc 1GiB
chunks to OOM, then 64MiB chunks. JSON to stdout."""
import torch, json

free0, total = torch.cuda.mem_get_info(0)
chunks, alloc = [], 0
GB, MB64 = 1 << 30, 64 * (1 << 20)
try:
    while True:
        chunks.append(torch.empty(GB, dtype=torch.uint8, device="cuda:0"))
        alloc += GB
except torch.cuda.OutOfMemoryError:
    pass
try:
    while True:
        chunks.append(torch.empty(MB64, dtype=torch.uint8, device="cuda:0"))
        alloc += MB64
except torch.cuda.OutOfMemoryError:
    pass
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "mem_get_info_total_gib": total / GB,
    "mem_get_info_free_at_start_gib": free0 / GB,
    "allocated_gib": alloc / GB,
    "missing_gib": total / GB - alloc / GB,
}, indent=1))

#!/usr/bin/env python3
"""CPU<->GPU and GPU<->GPU copy bandwidth/latency vs transfer size.

Probes the CC bounce-buffer path (SWIOTLB + GPU AES-GCM): pinned vs pageable,
pipelined (sync at end) vs synced (per-op round trip). JSON to stdout.
"""
import torch, time, json

SIZES = [4096, 65536, 1048576, 16777216, 268435456, 1073741824]

def iters_for(n):
    if n <= 1048576: return 200
    if n <= 16777216: return 30
    if n <= 268435456: return 8
    return 4

def bench_copy(src, dst, iters, sync_each=False):
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=not sync_each)
        if sync_each:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

out = {"info": {"device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device_count": torch.cuda.device_count()},
       "pinned_alloc_s": {}, "h2d": [], "d2h": [], "d2d": []}

for pinned in (True, False):
    for n in SIZES:
        t0 = time.perf_counter()
        host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
        if pinned:
            out["pinned_alloc_s"][str(n)] = time.perf_counter() - t0
        devt = torch.empty(n, dtype=torch.uint8, device="cuda:0")
        it = iters_for(n)
        for direction, s, d in (("h2d", host, devt), ("d2h", devt, host)):
            dt_pipe = bench_copy(s, d, it)
            dt_sync = bench_copy(s, d, min(it, 50), sync_each=True)
            out[direction].append({
                "bytes": n, "pinned": pinned,
                "lat_pipelined_us": dt_pipe * 1e6,
                "lat_synced_us": dt_sync * 1e6,
                "gbps_pipelined": n / dt_pipe / 1e9})
        del host, devt
        torch.cuda.empty_cache()

# GPU0 -> GPU1 direct copy: P2P over NVLink if enabled, else host-staged
if torch.cuda.device_count() > 1:
    out["info"]["can_access_peer_0_1"] = torch.cuda.can_device_access_peer(0, 1)
    for n in SIZES:
        a = torch.empty(n, dtype=torch.uint8, device="cuda:0")
        b = torch.empty(n, dtype=torch.uint8, device="cuda:1")
        dt = bench_copy(a, b, iters_for(n))
        out["d2d"].append({"bytes": n, "lat_pipelined_us": dt * 1e6,
                           "gbps": n / dt / 1e9})
        del a, b
        torch.cuda.empty_cache()

print(json.dumps(out, indent=1))

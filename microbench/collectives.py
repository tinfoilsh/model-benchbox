#!/usr/bin/env python3
"""NCCL all-reduce (TP proxy) and all-to-all (EP proxy) latency/bandwidth vs
message size, 8 GPUs. Run under: torchrun --standalone --nproc-per-node=8.
Rank 0 prints JSON to stdout. NCCL_DEBUG=INFO log captures topology selection
(P2P/NVLink vs SHM/host-staged) — gotcha #7 in HANDOFF.md.
"""
import torch, torch.distributed as dist, time, json

SIZES = [4096, 65536, 1048576, 8388608, 67108864, 268435456]

def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

res = {"world": world, "nccl": list(torch.cuda.nccl.version()),
       "allreduce": [], "alltoall": []}

for n in SIZES:
    x = torch.empty(n // 2, dtype=torch.float16, device="cuda")
    it = 50 if n <= 8388608 else 10
    dt = bench(lambda: dist.all_reduce(x), it)
    if rank == 0:
        algbw = n / dt / 1e9
        res["allreduce"].append({"bytes": n, "lat_us": dt * 1e6,
                                 "algbw_gbps": algbw,
                                 "busbw_gbps": algbw * 2 * (world - 1) / world})
    del x

for n in SIZES:
    elems = (n // 2) - (n // 2) % world  # equal splits across ranks
    a = torch.empty(elems, dtype=torch.float16, device="cuda")
    b = torch.empty(elems, dtype=torch.float16, device="cuda")
    it = 50 if n <= 8388608 else 10
    dt = bench(lambda: dist.all_to_all_single(b, a), it)
    if rank == 0:
        res["alltoall"].append({"bytes": n, "lat_us": dt * 1e6,
                                "algbw_gbps": n / dt / 1e9})
    del a, b

if rank == 0:
    print(json.dumps(res, indent=1))
dist.destroy_process_group()

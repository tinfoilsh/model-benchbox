#!/usr/bin/env python3
"""On-GPU HBM memory-bandwidth probe (STREAM-like), CC on vs off.

Tests whether the unattributed per-token decode tax is an HBM-bandwidth tax:
decode streams weights+KV from HBM every token, but our graph-replay microbench
used trivial L2-resident kernels and saw 'free'. This measures actual on-GPU
memory bandwidth (no host/PCIe involved) three ways:

 (1) eager bandwidth (copy/triad/reduce) at HBM-bound sizes 256MB-2GB,
 (2) graph-replay copy bandwidth (LAUNCH-FREE) at 1MB (L2-resident) vs 1GB
     (HBM-bound) -- the clean L2-vs-HBM discriminator; if CC taxes only the
     1GB point, the cost is HBM bandwidth, and it explains why (1)-the trivial
     replay looked free,
 (3) compute-bound GEMM TFLOPS -- the memory-vs-compute discriminator; should
     be ~free under CC if the tax is memory, not compute.

Run identically under CC-on and CC-off; the on/off ratio at HBM sizes is the
answer. JSON to stdout.
"""
import torch, time, json

dev = "cuda:0"
torch.zeros(1, device=dev); torch.cuda.synchronize()  # init context
out = {"device": torch.cuda.get_device_name(0),
       "torch": torch.__version__, "cuda": torch.version.cuda}


def eager_bw(nbytes, op, rw, iters=50):
    n = nbytes // 4
    a = torch.randn(n, device=dev); b = torch.randn(n, device=dev); c = torch.empty(n, device=dev)
    for _ in range(5): op(a, b, c)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): op(a, b, c)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    g = rw * nbytes / dt / 1e9
    del a, b, c; torch.cuda.empty_cache()
    return round(g, 1)


OPS = {"copy":   (lambda a, b, c: c.copy_(a), 2),
       "triad":  (lambda a, b, c: torch.add(a, b, alpha=2.0, out=c), 3),
       "reduce": (lambda a, b, c: a.sum(), 1)}

out["eager_bw_gbps"] = {}
for nm, (op, rw) in OPS.items():
    row = {}
    for s_mb in [256, 1024, 2048]:
        try: row[f"{s_mb}MB"] = eager_bw(s_mb << 20, op, rw)
        except RuntimeError as e: row[f"{s_mb}MB"] = f"ERR {str(e)[:30]}"
    out["eager_bw_gbps"][nm] = row


def graph_copy_bw(nbytes, K=20, reps=50):
    n = nbytes // 4
    a = torch.randn(n, device=dev); c = torch.empty(n, device=dev)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3): c.copy_(a)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        for _ in range(K): c.copy_(a)
    gph.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): gph.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps / K
    g = 2 * nbytes / dt / 1e9
    del a, c, gph; torch.cuda.empty_cache()
    return round(g, 1)


out["graph_replay_copy_bw_gbps"] = {"1MB_L2": graph_copy_bw(1 << 20),
                                    "1GB_HBM": graph_copy_bw(1 << 30)}


def gemm_tflops(n=8192, iters=30):
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    for _ in range(5): torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): torch.mm(a, b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return round(2 * n ** 3 / dt / 1e12, 1)


out["gemm_fp16_tflops_8192"] = gemm_tflops()
print(json.dumps(out, indent=1))

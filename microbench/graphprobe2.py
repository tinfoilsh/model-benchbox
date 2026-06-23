#!/usr/bin/env python3
"""Phase-resolved CUDA-graph control-path probe.

Separately times capture+instantiate vs graph DESTROY (teardown), cold vs
steady, at 1 and 50 kernels. membench3.py bundles all three into one mean-of-10
(tiny_graph_capture_instantiate_ms) and graphprobe.py excluded destroy — so
neither could tell whether a (multi-GPU?) CC graph toll lives in instantiate or
in per-peer GSP teardown. This separates them. Reports device + GPU count so the
1-GPU vs 8-GPU/NVLE comparison is unambiguous. JSON to stdout.
"""
import torch, time, json, statistics

dev = "cuda:0"
x = torch.ones(64, device=dev)
out = {"device": torch.cuda.get_device_name(0),
       "ngpu": torch.cuda.device_count(),
       "torch": torch.__version__, "cuda": torch.version.cuda}


def phases(nkern):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(nkern):
            x.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # capture + instantiate
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        for _ in range(nkern):
            x.add_(1.0)
    torch.cuda.synchronize()
    t_cap = (time.perf_counter() - t0) * 1e3
    # destroy / teardown
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if hasattr(g, "reset"):
        g.reset()
    del g
    torch.cuda.synchronize()
    t_destroy = (time.perf_counter() - t0) * 1e3
    return t_cap, t_destroy


x.add_(1.0); torch.cuda.synchronize()  # context + kernel live before graph work

for nkern in [1, 50]:
    cold = phases(nkern)
    series = [phases(nkern) for _ in range(30)]
    caps = [c for c, d in series]
    dess = [d for c, d in series]
    out[f"graph_{nkern}kern"] = {
        "cap_inst_cold_ms": round(cold[0], 3),
        "cap_inst_steady_median_ms": round(statistics.median(caps), 3),
        "cap_inst_steady_max_ms": round(max(caps), 3),
        "destroy_cold_ms": round(cold[1], 3),
        "destroy_steady_median_ms": round(statistics.median(dess), 3),
        "destroy_steady_max_ms": round(max(dess), 3),
        "n": len(series),
    }

print(json.dumps(out, indent=1))

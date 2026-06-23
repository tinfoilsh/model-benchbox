#!/usr/bin/env python3
"""Focused probe — is the Blackwell-CC graph capture+instantiate toll a FIXED
PER-GRAPH cost, or a one-time first-call subsystem init?

membench3.py's control_path cell reports only the MEAN of 10 captures
(tiny_graph_capture_instantiate_ms), which cannot distinguish:
  (A) "~100 ms x every graph"      -> compounds at startup, breaks DP+EP   [the claim]
  (B) "~1 s once, then cheap"      -> benign one-time init                 [the alternative]

This logs the COLD first capture separately from a WARMED steady-state series,
so the two stories are no longer aliased. Also sweeps graph size (1 vs 10 vs 50
kernels) to see whether the toll is per-kernel (capture/record) or fixed
per-graph (instantiate). Pure torch/CUDA, no model, runs in seconds.
JSON to stdout.
"""
import torch, time, json, statistics

dev = "cuda:0"
x = torch.ones(64, device=dev)
out = {"device": torch.cuda.get_device_name(0),
       "torch": torch.__version__, "cuda": torch.version.cuda}


def capture_instantiate_ms(nkern):
    """One capture+instantiate of an nkern-kernel graph; kernels pre-warmed
    off-graph so we time the graph machinery, not kernel JIT."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(nkern):
            x.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        for _ in range(nkern):
            x.add_(1.0)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    del g
    return dt


# Make sure the CUDA context + the add_ kernel are live before any graph work,
# so 'cold' isolates the GRAPH subsystem's first-call cost, not context init.
x.add_(1.0); torch.cuda.synchronize()

for nkern in [1, 10, 50]:
    cold = capture_instantiate_ms(nkern)          # very first graph at this size
    series = [capture_instantiate_ms(nkern) for _ in range(30)]  # warmed steady state
    out[f"graph_{nkern}kern"] = {
        "cold_first_ms": round(cold, 3),
        "steady_median_ms": round(statistics.median(series), 3),
        "steady_min_ms": round(min(series), 3),
        "steady_max_ms": round(max(series), 3),
        "steady_n": len(series),
        "all_ms": [round(t, 2) for t in series],
        # the discriminator: cold/steady >> 1 => one-time init (benign);
        #                    cold/steady ~ 1  => fixed per-graph (the claim)
        "cold_over_steady": round(cold / statistics.median(series), 1),
    }

print(json.dumps(out, indent=1))

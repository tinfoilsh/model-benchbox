#!/usr/bin/env python3
"""Kernel-launch latency + CUDA graph capture/replay cost under CC.

Probes the encrypted command-submission channel (launches, doorbells, syncs)
and the graph-capture slowdown seen during the DP+EP attempt (~38s/graph vs
~6s). JSON to stdout.
"""
import torch, time, json

x = torch.ones(256, device="cuda:0")
res = {}

def pipelined(n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1.0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

def roundtrip(n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1.0)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

for _ in range(3):
    pipelined(100)  # warmup / jit
res["launch_pipelined_us"] = pipelined(2000) * 1e6   # submission cost
res["launch_roundtrip_us"] = roundtrip(300) * 1e6    # launch+sync RTT

# CUDA graph: capture K small kernels, then replay
K, NREP = 300, 100
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):  # warmup on side stream (capture requirement)
    for _ in range(3):
        x.add_(1.0)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.cuda.graph(g):
    for _ in range(K):
        x.add_(1.0)
res["graph_capture_s"] = time.perf_counter() - t0
res["graph_capture_kernels"] = K
res["graph_capture_us_per_kernel"] = res["graph_capture_s"] / K * 1e6

g.replay()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(NREP):
    g.replay()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / NREP
res["graph_replay_us"] = dt * 1e6
res["graph_replay_us_per_kernel"] = dt / K * 1e6

print(json.dumps(res, indent=1))

#!/usr/bin/env python3
"""NVTX-labeled variant of launchbench.py for Nsight/strace attribution."""
import contextlib
import json
import os
import time

import torch


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


@contextlib.contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


PIPELINED_N = env_int("LAUNCHBENCH_PIPELINED_N", 2000)
ROUNDTRIP_N = env_int("LAUNCHBENCH_ROUNDTRIP_N", 300)
GRAPH_K = env_int("LAUNCHBENCH_GRAPH_K", 300)
GRAPH_REPS = env_int("LAUNCHBENCH_GRAPH_REPS", 100)
WARMUP_N = env_int("LAUNCHBENCH_WARMUP_N", 100)

x = torch.ones(256, device="cuda:0")
res = {
    "device": torch.cuda.get_device_name(0),
    "pipelined_n": PIPELINED_N,
    "roundtrip_n": ROUNDTRIP_N,
    "graph_kernels": GRAPH_K,
    "graph_reps": GRAPH_REPS,
}


def pipelined(n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx_range(f"launchbench:pipelined:n={n}"):
        for _ in range(n):
            x.add_(1.0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def roundtrip(n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with nvtx_range(f"launchbench:roundtrip:n={n}"):
        for _ in range(n):
            x.add_(1.0)
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


for _ in range(3):
    pipelined(WARMUP_N)

res["launch_pipelined_us"] = pipelined(PIPELINED_N) * 1e6
res["launch_roundtrip_us"] = roundtrip(ROUNDTRIP_N) * 1e6

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        x.add_(1.0)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

t0 = time.perf_counter()
with nvtx_range(f"launchbench:graph_capture:k={GRAPH_K}"):
    with torch.cuda.graph(g):
        for _ in range(GRAPH_K):
            x.add_(1.0)
res["graph_capture_s"] = time.perf_counter() - t0
res["graph_capture_us_per_kernel"] = res["graph_capture_s"] / GRAPH_K * 1e6

g.replay()
torch.cuda.synchronize()
t0 = time.perf_counter()
with nvtx_range(f"launchbench:graph_replay:reps={GRAPH_REPS}:k={GRAPH_K}"):
    for _ in range(GRAPH_REPS):
        g.replay()
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / GRAPH_REPS
res["graph_replay_us"] = dt * 1e6
res["graph_replay_us_per_kernel"] = dt / GRAPH_K * 1e6

print(json.dumps(res, indent=1))

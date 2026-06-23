#!/usr/bin/env python3
"""Pure-CPU bench: isolates source #4 (TDX CPU tax) — no GPU involved.
python loop, numpy matmul, memory bandwidth, sha256. JSON to stdout."""
import time, json, hashlib
import numpy as np

res = {}

t0 = time.perf_counter()
x = 0
for i in range(10_000_000):
    x += i
res["pyloop_Mops"] = 10 / (time.perf_counter() - t0)

a = np.random.rand(2048, 2048).astype(np.float32)
b = np.random.rand(2048, 2048).astype(np.float32)
a @ b
t0 = time.perf_counter()
for _ in range(10):
    a @ b
res["np_matmul_gflops"] = 10 * 2 * 2048**3 / (time.perf_counter() - t0) / 1e9

src = np.ones(1 << 30, dtype=np.uint8)
src.copy()
t0 = time.perf_counter()
for _ in range(5):
    dst = src.copy()
res["np_memcpy_GBps"] = 5 / (time.perf_counter() - t0)

buf = b"\xab" * (256 * 1024 * 1024)
hashlib.sha256(buf)
t0 = time.perf_counter()
for _ in range(4):
    hashlib.sha256(buf)
res["sha256_GBps"] = 1.0 / (time.perf_counter() - t0)

print(json.dumps(res, indent=1))

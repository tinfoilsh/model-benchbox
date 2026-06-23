#!/usr/bin/env python3
"""Single-GPU mechanism cells (round 3): sub-knee, multi-thread scaling,
D2H/NUMA multi-process, chunked staging, copies-in-graphs, control-path
costs, AES-GCM software baseline. Sections independent; errors recorded.
JSON to stdout."""
import torch, time, json, os, threading, subprocess, statistics

CH = 16 * 1024 * 1024
out = {"info": {"device": torch.cuda.get_device_name(0),
                "torch": torch.__version__}, "errors": {}}

def med3(fn):
    return statistics.median([fn() for _ in range(3)])

# --- S1: sub-512B knee, n=3 medians, sync + pipelined ----------------------
try:
    knee = []
    for n in (64, 256, 512, 1024, 4096, 16384):
        host = torch.empty(max(n, 1), dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(max(n, 1), dtype=torch.uint8, device="cuda:0")
        def sync_lat():
            for _ in range(5): dev.copy_(host)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                dev.copy_(host); torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 200 * 1e6
        def pipe_lat():
            for _ in range(5): dev.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(400): dev.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 400 * 1e6
        knee.append({"bytes": n, "sync_us": med3(sync_lat),
                     "pipelined_us": med3(pipe_lat)})
    out["subknee"] = knee
except Exception as e:
    out["errors"]["subknee"] = repr(e)

# --- S2: multi-THREAD single-process scaling --------------------------------
try:
    res = []
    for nt in (1, 2, 4, 8):
        bufs = [(torch.empty(CH, dtype=torch.uint8, pin_memory=True),
                 torch.empty(CH, dtype=torch.uint8, device="cuda:0"),
                 torch.cuda.Stream()) for _ in range(nt)]
        REP = 12
        barrier = threading.Barrier(nt + 1)
        def worker(i):
            h, d, s = bufs[i]
            with torch.cuda.stream(s):
                d.copy_(h, non_blocking=True)  # warmup
            torch.cuda.synchronize()
            barrier.wait()
            with torch.cuda.stream(s):
                for _ in range(REP):
                    d.copy_(h, non_blocking=True)
            s.synchronize()
            barrier.wait()
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(nt)]
        for t in ts: t.start()
        barrier.wait()                      # all warmed up
        t0 = time.perf_counter()
        barrier.wait()                      # all done copying
        dt = time.perf_counter() - t0
        for t in ts: t.join()
        res.append({"threads": nt, "agg_gbps": CH * nt * REP / dt / 1e9})
        del bufs; torch.cuda.empty_cache()
    out["multithread"] = res
except Exception as e:
    out["errors"]["multithread"] = repr(e)

# --- S5: chunked staging strategies for a 1GB transfer ----------------------
try:
    GB = 1 << 30
    res = {}
    src_pg = torch.empty(GB, dtype=torch.uint8)             # pageable source
    dev = torch.empty(GB, dtype=torch.uint8, device="cuda:0")
    # (a) direct pageable
    dev.copy_(src_pg); torch.cuda.synchronize()
    t0 = time.perf_counter(); dev.copy_(src_pg); torch.cuda.synchronize()
    res["direct_pageable_gbps"] = GB / (time.perf_counter() - t0) / 1e9
    # (b) direct pinned (alloc cost reported separately)
    t0 = time.perf_counter()
    src_pin = torch.empty(GB, dtype=torch.uint8, pin_memory=True)
    res["pinned_1g_alloc_s"] = time.perf_counter() - t0
    src_pin.copy_(src_pg)
    dev.copy_(src_pin); torch.cuda.synchronize()
    t0 = time.perf_counter(); dev.copy_(src_pin); torch.cuda.synchronize()
    res["direct_pinned_gbps"] = GB / (time.perf_counter() - t0) / 1e9
    del src_pin; torch.cuda.empty_cache()
    # (c) staged via reused 64MB pinned chunk (CPU copy + async H2D, 2 chunks ping-pong)
    CK = 64 * 1024 * 1024
    chunks = [torch.empty(CK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(GB // CK):
        c, s = chunks[i % 2], streams[i % 2]
        s.synchronize()
        c.copy_(src_pg[i*CK:(i+1)*CK])
        with torch.cuda.stream(s):
            dev[i*CK:(i+1)*CK].copy_(c, non_blocking=True)
    torch.cuda.synchronize()
    res["staged_64m_pingpong_gbps"] = GB / (time.perf_counter() - t0) / 1e9
    out["staging_1g"] = res
    del src_pg, dev, chunks; torch.cuda.empty_cache()
except Exception as e:
    out["errors"]["staging_1g"] = repr(e)

# --- S6: copies captured inside a CUDA graph --------------------------------
try:
    n = 1048576
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda:0")
    # eager pipelined reference
    for _ in range(5): dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / 200 * 1e6
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        dev.copy_(host, non_blocking=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    K = 50
    t0 = time.perf_counter()
    with torch.cuda.graph(g):
        for _ in range(K):
            dev.copy_(host, non_blocking=True)
    cap_s = time.perf_counter() - t0
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): g.replay()
    torch.cuda.synchronize()
    rep_us = (time.perf_counter() - t0) / 100 / K * 1e6
    out["copy_in_graph"] = {"eager_pipelined_us": eager_us,
                            "graph_capture_s": cap_s,
                            "graph_replay_us_per_copy": rep_us,
                            "copy_bytes": n, "copies_per_graph": K}
except Exception as e:
    out["errors"]["copy_in_graph"] = repr(e)

# --- S7: control-path op costs ----------------------------------------------
try:
    res = {}
    t0 = time.perf_counter()
    for _ in range(20):
        x = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True); del x
    res["mallochost_64m_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    for _ in range(20):
        x = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        del x
        torch.cuda.empty_cache()   # force real cudaFree/cudaMalloc each iter
    res["cudamalloc_64m_ms"] = (time.perf_counter() - t0) / 20 * 1e3
    ev = [torch.cuda.Event() for _ in range(2)]
    t0 = time.perf_counter()
    for _ in range(2000):
        ev[0].record(); ev[1].record()
    torch.cuda.synchronize()
    res["event_record_us"] = (time.perf_counter() - t0) / 4000 * 1e6
    # tiny-graph instantiate cost
    xs = torch.ones(64, device="cuda:0")
    t0 = time.perf_counter()
    for _ in range(10):
        gg = torch.cuda.CUDAGraph()
        ss = torch.cuda.Stream()
        with torch.cuda.stream(ss):
            xs.add_(1.0)
        torch.cuda.current_stream().wait_stream(ss)
        with torch.cuda.graph(gg):
            xs.add_(1.0)
        del gg
    res["tiny_graph_capture_instantiate_ms"] = (time.perf_counter() - t0) / 10 * 1e3
    out["control_path"] = res
except Exception as e:
    out["errors"]["control_path"] = repr(e)

# --- S8: software AES-256-GCM single-core baseline ---------------------------
try:
    p = subprocess.run(["openssl", "speed", "-evp", "aes-256-gcm", "-seconds", "1"],
                       capture_output=True, text=True, timeout=60)
    line = [l for l in p.stdout.splitlines() if "aes-256-gcm" in l.lower()]
    out["openssl_gcm"] = line[-1] if line else p.stdout[-300:]
except Exception as e:
    out["errors"]["openssl_gcm"] = repr(e)

print(json.dumps(out, indent=1))

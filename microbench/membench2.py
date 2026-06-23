#!/usr/bin/env python3
"""Round 2: knee sweep, multi-stream + multi-process H2D scaling, copy/compute
overlap, CPU-cycles-per-byte. Fills the §7 #7 'still open' items. JSON to stdout.
Each section is independent — failures recorded, rest continues.
"""
import torch, time, json, os

out = {"info": {"device": torch.cuda.get_device_name(0),
                "torch": torch.__version__}, "errors": {}}

# --- 1) knee sweep: fine-grained small sizes, sync-each latency -------------
try:
    knee = []
    for pinned in (True, False):
        for n in (512, 2048, 4096, 8192, 16384, 65536, 262144):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            dev = torch.empty(n, dtype=torch.uint8, device="cuda:0")
            for _ in range(5):
                dev.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                dev.copy_(host)
                torch.cuda.synchronize()
            knee.append({"bytes": n, "pinned": pinned,
                         "lat_us": (time.perf_counter() - t0) / 200 * 1e6})
    out["knee"] = knee
except Exception as e:
    out["errors"]["knee"] = repr(e)

# --- 2) multi-stream scaling (single process/thread) ------------------------
CH = 16 * 1024 * 1024
try:
    ms = []
    for pinned in (True, False):
        for ns in (1, 2, 4, 8):
            streams = [torch.cuda.Stream() for _ in range(ns)]
            hosts = [torch.empty(CH, dtype=torch.uint8, pin_memory=pinned)
                     for _ in range(ns)]
            devs = [torch.empty(CH, dtype=torch.uint8, device="cuda:0")
                    for _ in range(ns)]
            REP = 8
            for s, h, d in zip(streams, hosts, devs):  # warmup
                with torch.cuda.stream(s):
                    d.copy_(h, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REP):
                for s, h, d in zip(streams, hosts, devs):
                    with torch.cuda.stream(s):
                        d.copy_(h, non_blocking=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ms.append({"pinned": pinned, "streams": ns,
                       "agg_gbps": CH * ns * REP / dt / 1e9})
            del streams, hosts, devs
            torch.cuda.empty_cache()
    out["multistream"] = ms
except Exception as e:
    out["errors"]["multistream"] = repr(e)

# --- 3) copy/compute overlap -------------------------------------------------
# t_both ~= max(t_mm, t_cp) => overlapped; ~= t_mm + t_cp => serialized.
try:
    N = 8192
    A = torch.randn(N, N, device="cuda:0", dtype=torch.float16)
    B = torch.randn(N, N, device="cuda:0", dtype=torch.float16)
    CPB = 64 * 1024 * 1024
    hpin = torch.empty(CPB, dtype=torch.uint8, pin_memory=True)
    dbuf = torch.empty(CPB, dtype=torch.uint8, device="cuda:0")
    mm_s, cp_s = torch.cuda.Stream(), torch.cuda.Stream()
    MM, CP = 200, 64

    def run(do_mm, do_cp):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if do_mm:
            with torch.cuda.stream(mm_s):
                for _ in range(MM):
                    torch.mm(A, B)
        if do_cp:
            with torch.cuda.stream(cp_s):
                for _ in range(CP):
                    dbuf.copy_(hpin, non_blocking=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    run(True, False)  # warmup
    t_mm = run(True, False)
    t_cp = run(False, True)
    t_both = run(True, True)
    saved = t_mm + t_cp - t_both
    out["overlap"] = {"t_mm_s": t_mm, "t_cp_s": t_cp, "t_both_s": t_both,
                      "overlap_efficiency": saved / min(t_mm, t_cp),
                      "mm_iters": MM, "copies": CP, "copy_bytes": CPB}
    del A, B, hpin, dbuf
    torch.cuda.empty_cache()
except Exception as e:
    out["errors"]["overlap"] = repr(e)

# --- 4) CPU time per byte copied (encrypt cycles show up here) ---------------
try:
    cpb = []
    for pinned in (True, False):
        host = torch.empty(CH, dtype=torch.uint8, pin_memory=pinned)
        dev = torch.empty(CH, dtype=torch.uint8, device="cuda:0")
        dev.copy_(host)
        torch.cuda.synchronize()
        w0, c0 = time.perf_counter(), time.process_time()
        REP = 64
        for _ in range(REP):
            dev.copy_(host)
            torch.cuda.synchronize()
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        gb = CH * REP / 1e9
        cpb.append({"pinned": pinned, "wall_s_per_gb": wall / gb,
                    "cpu_s_per_gb": cpu / gb})
    out["cpu_per_byte"] = cpb
except Exception as e:
    out["errors"]["cpu_per_byte"] = repr(e)

print(json.dumps(out, indent=1))

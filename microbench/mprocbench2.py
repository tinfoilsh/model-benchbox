#!/usr/bin/env python3
"""Multi-process copy scaling, round 2: D2H direction + core-pinned H2D.
JSON to stdout."""
import torch, torch.multiprocessing as mp, time, json, os

CH = 16 * 1024 * 1024
REP = 24

def worker(rank, barrier, q, direction, pin_cores):
    try:
        if pin_cores:
            os.sched_setaffinity(0, {rank * 2, rank * 2 + 1})
        torch.cuda.set_device(0)
        host = torch.empty(CH, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(CH, dtype=torch.uint8, device="cuda:0")
        src, dst = (dev, host) if direction == "d2h" else (host, dev)
        dst.copy_(src)
        torch.cuda.synchronize()
        barrier.wait(timeout=120)
        t0 = time.perf_counter()
        for _ in range(REP):
            dst.copy_(src)
            torch.cuda.synchronize()
        q.put({"rank": rank, "gbps": CH * REP / (time.perf_counter() - t0) / 1e9})
    except Exception as e:
        q.put({"rank": rank, "error": repr(e)})

def cell(nproc, direction, pin_cores):
    barrier = mp.Barrier(nproc)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, barrier, q, direction, pin_cores))
             for r in range(nproc)]
    for p in procs: p.start()
    outs = []
    for _ in range(nproc):
        try: outs.append(q.get(timeout=180))
        except Exception: outs.append({"error": "queue timeout"})
    for p in procs:
        p.join(timeout=30)
        if p.is_alive(): p.terminate()
    ok = [o for o in outs if "gbps" in o]
    return {"procs": nproc, "direction": direction, "pinned_cores": pin_cores,
            "ok": len(ok), "agg_gbps": sum(o["gbps"] for o in ok) if ok else None}

def main():
    mp.set_start_method("spawn", force=True)
    res = {"chunk_bytes": CH, "rep": REP, "cells": []}
    for d in ("d2h", "h2d"):
        for np_ in (1, 2, 4, 8):
            res["cells"].append(cell(np_, d, False))
    for np_ in (4, 8):                       # pinned-core variant, h2d
        res["cells"].append(cell(np_, "h2d", True))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()

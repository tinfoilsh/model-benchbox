#!/usr/bin/env python3
"""Multi-PROCESS pinned H2D scaling: does the encrypted-copy path scale with
CPU cores (per-core SW-crypto bound) where multi-stream (single thread) can't?
Spawn N procs, each copies pinned 16MB x REP after a barrier. JSON to stdout.
"""
import torch, torch.multiprocessing as mp, time, json, sys

CH = 16 * 1024 * 1024
REP = 24

def worker(rank, barrier, q):
    try:
        torch.cuda.set_device(0)
        host = torch.empty(CH, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(CH, dtype=torch.uint8, device="cuda:0")
        dev.copy_(host)
        torch.cuda.synchronize()
        barrier.wait(timeout=120)
        t0 = time.perf_counter()
        for _ in range(REP):
            dev.copy_(host)
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        q.put({"rank": rank, "gbps": CH * REP / dt / 1e9, "secs": dt})
    except Exception as e:
        q.put({"rank": rank, "error": repr(e)})

def main():
    mp.set_start_method("spawn", force=True)
    res = {"chunk_bytes": CH, "rep": REP, "runs": []}
    for nproc in (1, 2, 4, 8):
        barrier = mp.Barrier(nproc)
        q = mp.Queue()
        procs = [mp.Process(target=worker, args=(r, barrier, q))
                 for r in range(nproc)]
        for p in procs:
            p.start()
        outs = []
        for _ in range(nproc):
            try:
                outs.append(q.get(timeout=180))
            except Exception:
                outs.append({"error": "queue timeout"})
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        ok = [o for o in outs if "gbps" in o]
        res["runs"].append({
            "procs": nproc, "ok": len(ok),
            "agg_gbps": sum(o["gbps"] for o in ok) if ok else None,
            "per_proc_gbps": [round(o["gbps"], 2) for o in ok]})
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe CUDA CPU/GPU boundary costs relevant to vLLM decode streaming.

This benchmark intentionally measures CPU wall time around small CUDA runtime
operations, because the Gemma/H200 traces showed large runtime API time for tiny
payloads while copy-engine activity stayed small.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import Callable

import torch


SLEEP_CYCLES = 0


def us(ns: int) -> float:
    return ns / 1000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(values) if values else 0.0,
        "median_us": statistics.median(values) if values else 0.0,
        "p90_us": percentile(values, 0.90),
        "p99_us": percentile(values, 0.99),
        "min_us": min(values) if values else 0.0,
        "max_us": max(values) if values else 0.0,
    }


def measure_block(
    name: str,
    fn: Callable[[], None],
    *,
    repeats: int,
    warmup: int,
    sync_after: bool = True,
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    enqueue_samples: list[float] = []
    total_samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        if sync_after:
            torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        enqueue_samples.append(us(t1 - t0))
        total_samples.append(us(t2 - t0))
    out = {
        "name": name,
        "repeats": repeats,
        "sync_after": sync_after,
    }
    out.update({f"enqueue_{k}": v for k, v in summarize(enqueue_samples).items()})
    out.update({f"total_{k}": v for k, v in summarize(total_samples).items()})
    return out


def measure_loop(
    name: str,
    fn: Callable[[], None],
    *,
    iters: int,
    warmup: int,
    sync_after: bool = True,
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter_ns()
    if sync_after:
        torch.cuda.synchronize()
    t2 = time.perf_counter_ns()
    return {
        "name": name,
        "iters": iters,
        "sync_after": sync_after,
        "enqueue_us_per_iter": us(t1 - t0) / iters,
        "total_us_per_iter": us(t2 - t0) / iters,
    }


def tiny_gpu_work(buf: torch.Tensor, repeats: int) -> None:
    if SLEEP_CYCLES > 0 and hasattr(torch.cuda, "_sleep"):
        for _ in range(repeats):
            torch.cuda._sleep(SLEEP_CYCLES)
        return
    for _ in range(repeats):
        buf.add_(1)


def copy_microbench(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sizes = [int(x) for x in args.sizes.split(",") if x]
    max_ints = max(1, max(sizes) // 4)
    dev = torch.arange(max_ints, device="cuda", dtype=torch.int32)
    work = torch.ones(1024, device="cuda", dtype=torch.float32)

    for size in sizes:
        elems = max(1, size // 4)
        for pinned in (False, True):
            host = torch.empty(elems, device="cpu", dtype=torch.int32, pin_memory=pinned)

            def d2h_ready() -> None:
                host.copy_(dev[:elems], non_blocking=True)

            def d2h_after_work_sync_each() -> None:
                tiny_gpu_work(work, args.kernel_repeats)
                host.copy_(dev[:elems], non_blocking=True)
                torch.cuda.current_stream().synchronize()

            def d2h_after_work_pipelined() -> None:
                tiny_gpu_work(work, args.kernel_repeats)
                host.copy_(dev[:elems], non_blocking=True)

            rows.append({
                "group": "d2h",
                "size_bytes": size,
                "pinned": pinned,
                **measure_loop(
                    "ready_device_to_host_pipelined",
                    d2h_ready,
                    iters=args.iters,
                    warmup=args.warmup,
                ),
            })
            rows.append({
                "group": "d2h",
                "size_bytes": size,
                "pinned": pinned,
                **measure_loop(
                    "gpu_work_then_d2h_sync_each",
                    d2h_after_work_sync_each,
                    iters=args.sync_iters,
                    warmup=max(1, args.warmup // 10),
                    sync_after=False,
                ),
            })
            rows.append({
                "group": "d2h",
                "size_bytes": size,
                "pinned": pinned,
                **measure_loop(
                    "gpu_work_then_d2h_pipelined",
                    d2h_after_work_pipelined,
                    iters=args.iters,
                    warmup=args.warmup,
                ),
            })

            h2d_src = torch.empty(elems, device="cpu", dtype=torch.int32, pin_memory=pinned)
            h2d_dst = torch.empty(elems, device="cuda", dtype=torch.int32)

            def h2d_ready() -> None:
                h2d_dst.copy_(h2d_src, non_blocking=True)

            rows.append({
                "group": "h2d",
                "size_bytes": size,
                "pinned": pinned,
                **measure_loop(
                    "host_to_device_pipelined",
                    h2d_ready,
                    iters=args.iters,
                    warmup=args.warmup,
                ),
            })
    return rows


def many_vs_packed(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    size = args.pack_size
    elems = max(1, size // 4)
    copies = args.pack_copies
    for pinned in (False, True):
        h_srcs = [
            torch.empty(elems, device="cpu", dtype=torch.int32, pin_memory=pinned)
            for _ in range(copies)
        ]
        d_dsts = [torch.empty(elems, device="cuda", dtype=torch.int32) for _ in range(copies)]
        d_srcs = [torch.empty(elems, device="cuda", dtype=torch.int32) for _ in range(copies)]
        h_dsts = [
            torch.empty(elems, device="cpu", dtype=torch.int32, pin_memory=pinned)
            for _ in range(copies)
        ]
        h_pack_src = torch.empty(copies * elems, device="cpu", dtype=torch.int32, pin_memory=pinned)
        d_pack_dst = torch.empty(copies * elems, device="cuda", dtype=torch.int32)
        d_pack_src = torch.empty(copies * elems, device="cuda", dtype=torch.int32)
        h_pack_dst = torch.empty(copies * elems, device="cpu", dtype=torch.int32, pin_memory=pinned)

        def h2d_many() -> None:
            for src, dst in zip(h_srcs, d_dsts):
                dst.copy_(src, non_blocking=True)

        def h2d_packed() -> None:
            d_pack_dst.copy_(h_pack_src, non_blocking=True)

        def d2h_many() -> None:
            for src, dst in zip(d_srcs, h_dsts):
                dst.copy_(src, non_blocking=True)

        def d2h_packed() -> None:
            h_pack_dst.copy_(d_pack_src, non_blocking=True)

        for group, name, fn in (
            ("h2d_pack", "many_small", h2d_many),
            ("h2d_pack", "one_packed", h2d_packed),
            ("d2h_pack", "many_small", d2h_many),
            ("d2h_pack", "one_packed", d2h_packed),
        ):
            rows.append({
                "group": group,
                "size_bytes_each": size,
                "copies": copies,
                "total_bytes": size * copies,
                "pinned": pinned,
                **measure_loop(name, fn, iters=args.iters, warmup=args.warmup),
            })
    return rows


def event_microbench(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = torch.ones(1024, device="cuda", dtype=torch.float32)
    event = torch.cuda.Event(blocking=False, enable_timing=False)
    host = torch.empty(1, device="cpu", dtype=torch.int32, pin_memory=True)
    token = torch.ones(1, device="cuda", dtype=torch.int32)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    def record_after_work() -> None:
        tiny_gpu_work(work, args.kernel_repeats)
        event.record(torch.cuda.current_stream())

    def side_stream_wait_copy_pipelined() -> None:
        with torch.cuda.stream(producer):
            tiny_gpu_work(work, args.kernel_repeats)
            event.record(producer)
        with torch.cuda.stream(consumer):
            consumer.wait_event(event)
            host.copy_(token, non_blocking=True)

    def side_stream_wait_copy_sync_each() -> None:
        side_stream_wait_copy_pipelined()
        consumer.synchronize()

    rows.append({
        "group": "event",
        **measure_loop(
            "event_record_after_gpu_work",
            record_after_work,
            iters=args.iters,
            warmup=args.warmup,
        ),
    })
    rows.append({
        "group": "event",
        **measure_loop(
            "producer_event_consumer_d2h_pipelined",
            side_stream_wait_copy_pipelined,
            iters=args.iters,
            warmup=args.warmup,
        ),
    })
    rows.append({
        "group": "event",
        **measure_loop(
            "producer_event_consumer_d2h_sync_each",
            side_stream_wait_copy_sync_each,
            iters=args.sync_iters,
            warmup=max(1, args.warmup // 10),
            sync_after=False,
        ),
    })
    return rows


def decode_chunking(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    steps = args.decode_steps
    batch = args.active_tokens_per_step
    ring = torch.zeros((steps, batch), device="cuda", dtype=torch.int32)
    host = torch.empty((steps, batch), device="cpu", dtype=torch.int32, pin_memory=True)
    src = torch.arange(batch, device="cuda", dtype=torch.int32)
    work = torch.ones(1024, device="cuda", dtype=torch.float32)

    def run_chunk(flush_every: int) -> None:
        for start in range(0, steps, flush_every):
            end = min(steps, start + flush_every)
            for step in range(start, end):
                tiny_gpu_work(work, args.kernel_repeats)
                ring[step].copy_(src, non_blocking=True)
            host[start:end].copy_(ring[start:end], non_blocking=True)
            torch.cuda.current_stream().synchronize()

    for flush_every in [int(x) for x in args.flush_every.split(",") if x]:
        if flush_every < 1:
            continue
        fn = lambda flush_every=flush_every: run_chunk(flush_every)
        row = measure_block(
            f"decode_flush_every_{flush_every}_steps",
            fn,
            repeats=args.chunk_repeats,
            warmup=max(1, args.warmup // 20),
            sync_after=False,
        )
        row.update({
            "group": "decode_chunking",
            "decode_steps": steps,
            "active_tokens_per_step": batch,
            "flush_every_steps": flush_every,
            "flushes": (steps + flush_every - 1) // flush_every,
            "bytes_per_step": batch * 4,
            "total_bytes": steps * batch * 4,
            "total_us_per_decode_step_mean": float(row["total_mean_us"]) / steps,
            "total_us_per_decode_step_median": float(row["total_median_us"]) / steps,
        })
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sync-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--kernel-repeats", type=int, default=1)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--sizes", default="4,16,32,64,256,1024,4096")
    parser.add_argument("--pack-size", type=int, default=32)
    parser.add_argument("--pack-copies", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--active-tokens-per-step", type=int, default=8)
    parser.add_argument("--flush-every", default="1,2,4,8,16,32,64,256")
    parser.add_argument("--chunk-repeats", type=int, default=30)
    parser.add_argument("--out")
    args = parser.parse_args()

    global SLEEP_CYCLES
    SLEEP_CYCLES = args.sleep_cycles

    torch.set_grad_enabled(False)
    torch.empty(1, device="cuda").fill_(1)
    torch.cuda.synchronize()

    result = {
        "schema": "boundary_tax_probe.v1",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0),
        "args": vars(args),
        "rows": [],
    }
    rows: list[dict[str, object]] = []
    rows.extend(copy_microbench(args))
    rows.extend(many_vs_packed(args))
    rows.extend(event_microbench(args))
    rows.extend(decode_chunking(args))
    result["rows"] = rows
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

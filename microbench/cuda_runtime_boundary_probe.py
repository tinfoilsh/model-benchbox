#!/usr/bin/env python3
"""Direct CUDA runtime boundary probe.

This intentionally avoids PyTorch tensor/copy APIs. It calls libcudart through
ctypes and times CUDA runtime calls around a queued device operation followed by
small host/device boundaries.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
CUDA_STREAM_NON_BLOCKING = 1
CUDA_EVENT_DISABLE_TIMING = 2


def load_cudart() -> ctypes.CDLL:
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend([
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda-13.0/lib64/libcudart.so",
        "/usr/local/cuda-12.8/lib64/libcudart.so",
    ])
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("could not load libcudart: " + "; ".join(errors))


class Cuda:
    def __init__(self) -> None:
        self.lib = load_cudart()
        self._bind()

    def _bind(self) -> None:
        c_void_pp = ctypes.POINTER(ctypes.c_void_p)
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaStreamCreateWithFlags.argtypes = [c_void_pp, ctypes.c_uint]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.lib.cudaEventCreateWithFlags.argtypes = [c_void_pp, ctypes.c_uint]
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaMalloc.argtypes = [c_void_pp, ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaHostAlloc.argtypes = [c_void_pp, ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

    def check(self, err: int, name: str) -> None:
        if err != 0:
            msg = self.lib.cudaGetErrorString(err)
            detail = msg.decode("utf-8", "replace") if msg else "unknown"
            raise RuntimeError(f"{name} failed with {err}: {detail}")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self.check(self.lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        return int(count.value)

    def set_device(self, index: int) -> None:
        self.check(self.lib.cudaSetDevice(index), "cudaSetDevice")

    def stream(self) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreateWithFlags(ctypes.byref(out), CUDA_STREAM_NON_BLOCKING),
            "cudaStreamCreateWithFlags",
        )
        return out

    def event(self) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        self.check(
            self.lib.cudaEventCreateWithFlags(ctypes.byref(out), CUDA_EVENT_DISABLE_TIMING),
            "cudaEventCreateWithFlags",
        )
        return out

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(out), nbytes), "cudaMalloc")
        return out

    def host_alloc(self, nbytes: int) -> ctypes.c_void_p:
        out = ctypes.c_void_p()
        self.check(self.lib.cudaHostAlloc(ctypes.byref(out), nbytes, 0), "cudaHostAlloc")
        return out

    def sync_stream(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def sync_device(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def ptr_add(ptr: ctypes.c_void_p, offset: int) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(ptr.value) + offset)


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


def summarize_trials(trials: list[dict[str, float]]) -> dict[str, object]:
    keys = sorted({key for row in trials for key in row})
    return {key: summarize([row[key] for row in trials if key in row]) for key in keys}


class Buffers:
    def __init__(self, cuda: Cuda, work_bytes: int, copy_bytes: int) -> None:
        self.cuda = cuda
        alloc_bytes = max(1, work_bytes + copy_bytes + 4096)
        host_bytes = max(4096, copy_bytes)
        self.dev_a = cuda.malloc(alloc_bytes)
        self.dev_b = cuda.malloc(alloc_bytes)
        self.pinned_host = cuda.host_alloc(host_bytes)
        self.pageable_storage = ctypes.create_string_buffer(host_bytes)
        self.pageable_host = ctypes.c_void_p(ctypes.addressof(self.pageable_storage))

    def free(self) -> None:
        self.cuda.check(self.cuda.lib.cudaFree(self.dev_a), "cudaFree(dev_a)")
        self.cuda.check(self.cuda.lib.cudaFree(self.dev_b), "cudaFree(dev_b)")
        self.cuda.check(self.cuda.lib.cudaFreeHost(self.pinned_host), "cudaFreeHost")


def timed_trials(
    *,
    cuda: Cuda,
    repeats: int,
    warmup: int,
    run_once: Callable[[], dict[str, float]],
) -> list[dict[str, float]]:
    for _ in range(warmup):
        run_once()
    cuda.sync_device()
    trials = []
    for _ in range(repeats):
        trials.append(run_once())
    cuda.sync_device()
    return trials


def make_work(cuda: Cuda, buf: Buffers, stream: ctypes.c_void_p, work_bytes: int) -> None:
    if work_bytes > 0:
        cuda.check(
            cuda.lib.cudaMemsetAsync(buf.dev_a, 0x5A, work_bytes, stream),
            "cudaMemsetAsync(work)",
        )


def same_stream_case(
    cuda: Cuda,
    buf: Buffers,
    stream: ctypes.c_void_p,
    event: ctypes.c_void_p,
    *,
    op: str,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(stream)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, stream, work_bytes)
    t1 = time.perf_counter_ns()
    if op == "d2h":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                host_ptr, buf.dev_a, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream
            ),
            "cudaMemcpyAsync(D2H)",
        )
    elif op == "h2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, host_ptr, copy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream
            ),
            "cudaMemcpyAsync(H2D)",
        )
    elif op == "d2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, buf.dev_a, copy_bytes, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream
            ),
            "cudaMemcpyAsync(D2D)",
        )
    elif op == "event":
        cuda.check(cuda.lib.cudaEventRecord(event, stream), "cudaEventRecord")
    else:
        raise ValueError(op)
    t2 = time.perf_counter_ns()
    cuda.sync_stream(stream)
    t3 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "boundary_api_us": us(t2 - t1),
        "total_us": us(t3 - t0),
    }


def sync_then_d2h_case(
    cuda: Cuda,
    buf: Buffers,
    stream: ctypes.c_void_p,
    *,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(stream)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, stream, work_bytes)
    t1 = time.perf_counter_ns()
    cuda.sync_stream(stream)
    t2 = time.perf_counter_ns()
    cuda.check(
        cuda.lib.cudaMemcpyAsync(host_ptr, buf.dev_a, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream),
        "cudaMemcpyAsync(D2H-after-sync)",
    )
    t3 = time.perf_counter_ns()
    cuda.sync_stream(stream)
    t4 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "pre_sync_api_us": us(t2 - t1),
        "boundary_api_us": us(t3 - t2),
        "post_sync_api_us": us(t4 - t3),
        "total_us": us(t4 - t0),
    }


def side_stream_case(
    cuda: Cuda,
    buf: Buffers,
    producer: ctypes.c_void_p,
    consumer: ctypes.c_void_p,
    event: ctypes.c_void_p,
    *,
    op: str,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(producer)
    cuda.sync_stream(consumer)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, producer, work_bytes)
    t1 = time.perf_counter_ns()
    cuda.check(cuda.lib.cudaEventRecord(event, producer), "cudaEventRecord(producer)")
    t2 = time.perf_counter_ns()
    cuda.check(cuda.lib.cudaStreamWaitEvent(consumer, event, 0), "cudaStreamWaitEvent")
    t3 = time.perf_counter_ns()
    if op == "d2h":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                host_ptr, buf.dev_a, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, consumer
            ),
            "cudaMemcpyAsync(side D2H)",
        )
    elif op == "h2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, host_ptr, copy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, consumer
            ),
            "cudaMemcpyAsync(side H2D)",
        )
    elif op == "d2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, buf.dev_a, copy_bytes, CUDA_MEMCPY_DEVICE_TO_DEVICE, consumer
            ),
            "cudaMemcpyAsync(side D2D)",
        )
    else:
        raise ValueError(op)
    t4 = time.perf_counter_ns()
    cuda.sync_stream(consumer)
    t5 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "event_record_api_us": us(t2 - t1),
        "wait_event_api_us": us(t3 - t2),
        "boundary_api_us": us(t4 - t3),
        "total_us": us(t5 - t0),
    }


def same_stream_unrelated_case(
    cuda: Cuda,
    buf: Buffers,
    stream: ctypes.c_void_p,
    *,
    op: str,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(stream)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, stream, work_bytes)
    t1 = time.perf_counter_ns()
    if op == "d2h":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                host_ptr, buf.dev_b, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, stream
            ),
            "cudaMemcpyAsync(same-stream unrelated D2H)",
        )
    elif op == "h2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, host_ptr, copy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream
            ),
            "cudaMemcpyAsync(same-stream unrelated H2D)",
        )
    else:
        raise ValueError(op)
    t2 = time.perf_counter_ns()
    cuda.sync_stream(stream)
    t3 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "boundary_api_us": us(t2 - t1),
        "total_us": us(t3 - t0),
    }


def side_stream_unrelated_case(
    cuda: Cuda,
    buf: Buffers,
    producer: ctypes.c_void_p,
    consumer: ctypes.c_void_p,
    event: ctypes.c_void_p,
    *,
    op: str,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(producer)
    cuda.sync_stream(consumer)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, producer, work_bytes)
    t1 = time.perf_counter_ns()
    cuda.check(cuda.lib.cudaEventRecord(event, producer), "cudaEventRecord(unrelated producer)")
    t2 = time.perf_counter_ns()
    cuda.check(cuda.lib.cudaStreamWaitEvent(consumer, event, 0), "cudaStreamWaitEvent(unrelated)")
    t3 = time.perf_counter_ns()
    if op == "d2h":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                host_ptr, buf.dev_b, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, consumer
            ),
            "cudaMemcpyAsync(side unrelated D2H)",
        )
    elif op == "h2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, host_ptr, copy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, consumer
            ),
            "cudaMemcpyAsync(side unrelated H2D)",
        )
    else:
        raise ValueError(op)
    t4 = time.perf_counter_ns()
    cuda.sync_stream(consumer)
    t5 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "event_record_api_us": us(t2 - t1),
        "wait_event_api_us": us(t3 - t2),
        "boundary_api_us": us(t4 - t3),
        "total_us": us(t5 - t0),
    }


def independent_stream_case(
    cuda: Cuda,
    buf: Buffers,
    producer: ctypes.c_void_p,
    consumer: ctypes.c_void_p,
    *,
    op: str,
    host_ptr: ctypes.c_void_p,
    work_bytes: int,
    copy_bytes: int,
) -> dict[str, float]:
    cuda.sync_stream(producer)
    cuda.sync_stream(consumer)
    t0 = time.perf_counter_ns()
    make_work(cuda, buf, producer, work_bytes)
    t1 = time.perf_counter_ns()
    if op == "d2h":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                host_ptr, buf.dev_b, copy_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, consumer
            ),
            "cudaMemcpyAsync(independent D2H)",
        )
    elif op == "h2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, host_ptr, copy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, consumer
            ),
            "cudaMemcpyAsync(independent H2D)",
        )
    elif op == "d2d":
        cuda.check(
            cuda.lib.cudaMemcpyAsync(
                buf.dev_b, ptr_add(buf.dev_b, 2048), copy_bytes, CUDA_MEMCPY_DEVICE_TO_DEVICE, consumer
            ),
            "cudaMemcpyAsync(independent D2D)",
        )
    else:
        raise ValueError(op)
    t2 = time.perf_counter_ns()
    cuda.sync_stream(consumer)
    t3 = time.perf_counter_ns()
    cuda.sync_stream(producer)
    t4 = time.perf_counter_ns()
    return {
        "work_api_us": us(t1 - t0),
        "boundary_api_us": us(t2 - t1),
        "consumer_sync_us": us(t3 - t2),
        "producer_sync_us": us(t4 - t3),
        "total_us": us(t4 - t0),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    cuda = Cuda()
    count = cuda.device_count()
    cuda.set_device(args.device)
    rows = []
    for work_bytes in [int(x) for x in args.work_bytes.split(",") if x]:
        for copy_bytes in [int(x) for x in args.copy_bytes.split(",") if x]:
            buf = Buffers(cuda, work_bytes, copy_bytes)
            stream = cuda.stream()
            producer = cuda.stream()
            consumer = cuda.stream()
            event = cuda.event()
            try:
                for host_kind, host_ptr in (
                    ("pinned", buf.pinned_host),
                    ("pageable", buf.pageable_host),
                ):
                    host_filter = {x.strip() for x in args.host_kinds.split(",") if x.strip()}
                    if host_filter and host_kind not in host_filter:
                        continue
                    for op in ("d2h", "h2d", "d2d", "event"):
                        if op not in {"d2h", "h2d"} and host_kind != "pinned":
                            continue
                        if op in {"d2h", "h2d"}:
                            hp = host_ptr
                        else:
                            hp = buf.pinned_host
                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda op=op, hp=hp: same_stream_case(
                                cuda,
                                buf,
                                stream,
                                event,
                                op=op,
                                host_ptr=hp,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": f"same_stream_{op}",
                            "host_kind": host_kind if op in {"d2h", "h2d"} else "none",
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })

                    trials = timed_trials(
                        cuda=cuda,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        run_once=lambda hp=host_ptr: sync_then_d2h_case(
                            cuda,
                            buf,
                            stream,
                            host_ptr=hp,
                            work_bytes=work_bytes,
                            copy_bytes=copy_bytes,
                        ),
                    )
                    rows.append({
                        "scenario": "sync_then_d2h",
                        "host_kind": host_kind,
                        "work_bytes": work_bytes,
                        "copy_bytes": copy_bytes,
                        "summary": summarize_trials(trials),
                    })

                    for op in ("d2h", "h2d", "d2d"):
                        if op == "d2d" and host_kind != "pinned":
                            continue
                        hp = host_ptr if op in {"d2h", "h2d"} else buf.pinned_host
                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda op=op, hp=hp: side_stream_case(
                                cuda,
                                buf,
                                producer,
                                consumer,
                                event,
                                op=op,
                                host_ptr=hp,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": f"side_stream_{op}",
                            "host_kind": host_kind if op in {"d2h", "h2d"} else "none",
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })

                    for op in ("d2h", "h2d"):
                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda op=op, hp=host_ptr: same_stream_unrelated_case(
                                cuda,
                                buf,
                                stream,
                                op=op,
                                host_ptr=hp,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": f"same_stream_unrelated_{op}",
                            "host_kind": host_kind,
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })

                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda op=op, hp=host_ptr: side_stream_unrelated_case(
                                cuda,
                                buf,
                                producer,
                                consumer,
                                event,
                                op=op,
                                host_ptr=hp,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": f"side_stream_unrelated_{op}",
                            "host_kind": host_kind,
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })

                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda op=op, hp=host_ptr: independent_stream_case(
                                cuda,
                                buf,
                                producer,
                                consumer,
                                op=op,
                                host_ptr=hp,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": f"independent_stream_{op}",
                            "host_kind": host_kind,
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })

                    if host_kind == "pinned":
                        trials = timed_trials(
                            cuda=cuda,
                            repeats=args.repeats,
                            warmup=args.warmup,
                            run_once=lambda: independent_stream_case(
                                cuda,
                                buf,
                                producer,
                                consumer,
                                op="d2d",
                                host_ptr=host_ptr,
                                work_bytes=work_bytes,
                                copy_bytes=copy_bytes,
                            ),
                        )
                        rows.append({
                            "scenario": "independent_stream_d2d",
                            "host_kind": "none",
                            "work_bytes": work_bytes,
                            "copy_bytes": copy_bytes,
                            "summary": summarize_trials(trials),
                        })
            finally:
                cuda.check(cuda.lib.cudaEventDestroy(event), "cudaEventDestroy")
                cuda.check(cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy(stream)")
                cuda.check(cuda.lib.cudaStreamDestroy(producer), "cudaStreamDestroy(producer)")
                cuda.check(cuda.lib.cudaStreamDestroy(consumer), "cudaStreamDestroy(consumer)")
                buf.free()
    return {
        "schema": "cuda_runtime_boundary_probe.v2",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "device_count": count,
        "device": args.device,
        "args": vars(args),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--work-bytes", default="0,67108864,268435456")
    parser.add_argument("--copy-bytes", default="4,32")
    parser.add_argument("--host-kinds", default="pinned,pageable")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run_matrix(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Disable TensorRT-LLM v2 host KV-cache tier behind an environment switch.

The stock 1.3.0rc18 PyTorch backend auto-provisions a HostCacheTierConfig for
KVCacheManagerV2. NVIDIA CC rejects the host registration path with CUDA error
801, so Gemma4 hybrid attention cannot initialize. This patch keeps the stock
behavior unless TRTLLM_DISABLE_V2_HOST_KV_CACHE is enabled.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

SENTINEL = "TRTLLM_DISABLE_V2_HOST_KV_CACHE"

OLD = """        if host_quota > 0:
            cache_tiers.append(HostCacheTierConfig(quota=host_quota))
            logger.info(
                f"KV cache manager v2 host cache quota set to {host_quota / (1 << 30):.2f}GiB"
            )
"""

NEW = """        disable_host_kv_cache = os.environ.get(
            "TRTLLM_DISABLE_V2_HOST_KV_CACHE", "").lower() in (
                "1", "true", "yes", "on")
        if host_quota > 0 and not disable_host_kv_cache:
            cache_tiers.append(HostCacheTierConfig(quota=host_quota))
            logger.info(
                f"KV cache manager v2 host cache quota set to {host_quota / (1 << 30):.2f}GiB"
            )
        elif host_quota > 0:
            logger.warning(
                "KV cache manager v2 host cache disabled by "
                "TRTLLM_DISABLE_V2_HOST_KV_CACHE; GPU KV cache must be sized "
                "to avoid suspend/resume offload.")
"""


def tensorrt_llm_root() -> pathlib.Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("could not locate installed tensorrt_llm package")
    return pathlib.Path(next(iter(spec.submodule_search_locations)))


def patch_file(path: pathlib.Path) -> bool:
    if not path.exists():
        return False

    text = path.read_text()
    if SENTINEL in text:
        print(f"[patch] already patched: {path}")
        return True

    patched = text.replace(OLD, NEW, 1)
    if patched == text:
        return False

    path.write_text(patched)
    print(f"[patch] patched: {path}")
    return True


def main() -> int:
    root = tensorrt_llm_root()
    candidates = [
        root / "_torch" / "pyexecutor" / "resource_manager.py",
        root / "_torch" / "pyexecutor" / "kv_cache_manager_v2.py",
    ]

    patched = [path for path in candidates if patch_file(path)]
    if not patched:
        print("[patch] failed to find v2 host KV-cache tier block", file=sys.stderr)
        for path in candidates:
            print(f"[patch] checked: {path}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TRT-LLM Gemma benchbox.
#
# Gemma-4-31B forces TensorRT-LLM's v2 KV-cache manager. In stock rc18 that
# manager always creates a host KV offload tier, which calls CUDA host-memory
# registration and fails under NVIDIA confidential computing. This image keeps
# the CUDA-13.1 rc18 base that attests under cvm-version 0.10.4, but patches the
# Python v2 KV-cache setup so the host tier is skipped when
# TRTLLM_DISABLE_V2_HOST_KV_CACHE=1.

FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc18

ENV DEBIAN_FRONTEND=noninteractive
ENV TRTLLM_DISABLE_V2_HOST_KV_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git vim wget curl ca-certificates \
        openssh-client openssh-sftp-server \
        jq unzip tmux htop \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI is useful inside debug enclaves for pulling harness updates or
# publishing result bundles. The stock TRT-LLM image does not include it.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

COPY patches/disable_trtllm_v2_host_kv_cache.py /usr/local/bin/disable_trtllm_v2_host_kv_cache.py
RUN python3 /usr/local/bin/disable_trtllm_v2_host_kv_cache.py

WORKDIR /workspace

COPY CLAUDE.md /workspace/CLAUDE.md
COPY bench /workspace/bench

ENTRYPOINT []
CMD ["sleep", "infinity"]

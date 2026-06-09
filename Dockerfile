# Tinfoil model-benchbox: a deployable container with vLLM benchmarking tools
# baked in. Build context lives in this repo; tags with prefix `image-v*` push
# to ghcr.io/tinfoilsh/benchbox. The published digest is then pinned in
# tinfoil-config.yml and deployed as a Tinfoil Container.

# Pinned to v0.21.0 — the version the production confidential-kimi-k2-6-b200
# config is proven on. vLLM 0.22.1's image ships a broken nvidia-cutlass-dsl
# (mlir_global_dtors ICE) that crashes CUDA-graph capture for every NVFP4 MoE
# kernel path, so it cannot serve Kimi K2.6 NVFP4. Match production exactly.
ARG VLLM_VERSION=v0.21.0-ubuntu2404
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Match the dev-tool set that model_devbench installs on a debug enclave.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git vim wget curl ca-certificates \
        openssh-client openssh-sftp-server \
        jq unzip tmux htop \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI for clone/push from inside the enclave.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Bench tooling. The base image already ships vLLM with the
# `vllm bench {latency,serve,throughput}` subcommands, so we deliberately do NOT
# reinstall `vllm[bench]` here — that would pull the latest vLLM and clobber the
# pinned 0.21.0 base. We only add the client-side bench deps. guidellm is the
# vllm-project successor to benchmark_serving.py and our primary instrument.
RUN pip install --no-cache-dir \
        guidellm \
        openai \
        httpx \
        pandas \
        datasets

WORKDIR /workspace

COPY CLAUDE.md /workspace/CLAUDE.md

# CC-overhead benchmarking harness (serve a local engine + run guidellm/vllm
# bench over localhost). Version-controlled here so every release ships an
# auditable, attested copy of the exact experiment scripts.
COPY bench /workspace/bench

# Override the upstream OpenAI-server entrypoint so the container is a
# long-lived bench shell. SSH in via debug-mode and run benches interactively.
ENTRYPOINT []
CMD ["sleep", "infinity"]

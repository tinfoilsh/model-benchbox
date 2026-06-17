# Tinfoil model-benchbox: a deployable container with benchmarking tools baked
# in. The base remains the proven vLLM image because the official NeMo AutoModel
# container is too large for the GitHub Actions build runner; AutoModel is
# installed below for Gemma post-training workloads.
ARG VLLM_VERSION=v0.21.0-ubuntu2404
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Match the dev-tool set that model_devbench installs on a debug enclave.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git vim wget curl ca-certificates \
        openssh-client openssh-sftp-server \
        jq unzip tmux htop numactl \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI for clone/push from inside the enclave.
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Install AutoModel in the vLLM base instead of inheriting NVIDIA's full
# AutoModel container. Keep examples in /opt/Automodel so the runner can use the
# upstream VLM finetune entrypoint with a package version that matches it.
# The vLLM base has a Debian-owned blinker package without pip RECORD metadata;
# preinstall a pip-managed copy so Flask/mlflow dependency resolution can
# upgrade/satisfy it without tripping over uninstall-no-record-file.
RUN pip install --no-cache-dir --ignore-installed blinker

RUN git clone --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git /opt/Automodel \
    && pip install --no-cache-dir \
        "/opt/Automodel[cli,vlm]" \
        liger-kernel

WORKDIR /workspace

COPY CLAUDE.md /workspace/CLAUDE.md

# CC-overhead benchmarking harness (serve a local engine + run guidellm/vllm
# bench over localhost). Version-controlled here so every release ships an
# auditable, attested copy of the exact experiment scripts.
COPY bench /workspace/bench
COPY train /workspace/train

# Override the upstream entrypoint so the container is a long-lived bench shell.
# SSH in via debug-mode and run benches interactively.
ENTRYPOINT []
CMD ["sleep", "infinity"]

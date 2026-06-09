# CC-overhead bench harness

Measures NVIDIA confidential-computing overhead on inference by serving a model
as a **local in-process vLLM engine** inside the benchbox CVM and benchmarking it
over **localhost** — no TLS, no shim, no host networking — so the CC-on-vs-off
delta isn't confounded by the deployment path.

## Method

Deploy the **same** benchbox release twice on the GPU host, then run the same
script in each:

```bash
# CC ON  (TDX + GPU CC/PPCIe)
tinfoil container create benchkimi-cc-on  --repo tinfoilsh/model-benchbox --tag <tag> \
  --debug --host control.inf14.tinfoil.sh --ssh-key <key> \
  --secret GITHUB_TOKEN --secret HF_TOKEN --secret TINFOIL_API_KEY

# CC OFF (plain VM + GPU CC off) — same line plus:
  --disable-cc-mode --yes
```

> On this Intel/B300 host `--disable-cc-mode` flips **both** the GPU CC/PPCIe data
> path **and** CPU TDX memory encryption, so on-vs-off = full confidential-stack
> overhead. Isolating the GPU-only contribution needs a third arm
> (TDX-on + GPU-CC-off), not expressible via the product flag.

**Accessing the tools.** Debug SSH lands in the CVM *guest*, not the container.
The benchbox runs as a Docker container inside the guest, so step into it first:

```bash
ssh -A inf14.tinfoil.sh                       # agent-forward your org SSH key
ssh -p <ssh-port> root@localhost              # into the CVM guest
docker exec -it benchbox bash                 # into the benchbox container
```

Inside each enclave (one CVM holds all 8 GPUs, so run the arms sequentially):

```bash
cd /workspace/bench
./serve.sh                       # loads Kimi NVFP4 TP=8, waits for /health (~20 min)
./run_cc_experiment.sh cc-on     # or cc-off, matching how this CVM was deployed
```

> **Image is pinned to vLLM 0.21.0.** 0.22.1 ships a broken `nvidia-cutlass-dsl`
> that ICE-crashes CUDA-graph capture for every NVFP4 MoE kernel path, so it
> cannot serve Kimi K2.6 NVFP4. 0.21.0 is the version production runs it on.
> Scratch goes to `/dev/shm` (large via `ipc: host`); the container needs the
> `web` egress network for FlashInfer to fetch NVFP4 cubins at serve time.

Results land in `results/<cond>/<label>/` (raw guidellm + vllm-bench JSON, both
`/metrics` snapshots, `nvidia-smi` incl. CC-mode confirmation, `manifest.json`).
The CVM filesystem is a ramdisk — **save results out before teardown**.

## Operating points

| point | prompt / output | concurrency | regime |
|-------|-----------------|-------------|--------|
| short | 128 / 128       | 1, 4        | I/O-bound worst case (TTFT-dominated) |
| long  | 8192 / 512      | 16, 32      | compute-bound best case |

## Instruments

- **guidellm** (primary) — reports TTFT and ITL separately; `concurrent` rate type.
- **vllm bench serve** (cross-check) — independent confirmation of the same point.
- **/metrics + nvidia-smi** — server-side attribution + CC-mode verification.

Knobs via env: `MPK`, `PORT`, `MODEL`, `TOKENIZER`, `GUIDELLM_SECS`.
Eagle3 speculative decoding is intentionally **out** of this first baseline.

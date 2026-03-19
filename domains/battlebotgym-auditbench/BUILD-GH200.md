# AuditBench — Lambda GH200 Build Guide

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 80 GB | 96 GB (GH200) |
| System RAM | 64 GB | 96+ GB |
| Disk | 200 GB | 500+ GB |
| Architecture | x86_64 or ARM64 | ARM64 (GH200) |

**Why GH200?** Llama 3.3 70B with 8 concurrent LoRA adapters needs ~75 GB VRAM at fp16. The GH200's 96 GB unified memory handles this with headroom for `--gpu-memory-utilization 0.85`.

**Alternatives:** A100 80GB (tight), 8×A100 40GB (works but needs tensor parallelism), H100 80GB (overkill but works).

## Provider: Lambda Labs

1. Go to [cloud.lambda.ai](https://cloud.lambda.ai)
2. Launch **1× GH200** instance (cheapest option that fits 70B + LoRAs)
3. Add your SSH key during setup
4. Note the IP address once provisioned

## Setup (15 minutes)

### Step 1: Transfer battleBOT

battleBOT is local-only (not on GitHub). Transfer via scp:

```bash
# From your local machine:
cd ~/Downloads
tar czf battleBOT.tar.gz battleBOT/
scp battleBOT.tar.gz ubuntu@<IP>:~/
ssh ubuntu@<IP> "tar xzf battleBOT.tar.gz && rm battleBOT.tar.gz"
```

### Step 2: Run setup script

```bash
ssh ubuntu@<IP>
bash ~/battleBOT/games/auditbench/setup-lambda.sh
```

This installs:
- Python 3.11 via `uv` (system Python is 3.10, auditing-agents needs 3.11)
- PyTorch with CUDA (ARM64 cu128 wheels)
- vLLM (standard, not steer fork — fork doesn't build on ARM64)
- auditing-agents + safety-tooling from GitHub
- Claude CLI

### Step 3: HuggingFace authentication

Llama 3.3 70B is a gated model. You need:
1. HuggingFace account with access granted at [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
2. HF token set on the instance:

```bash
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN" > ~/.cache/huggingface/token
echo 'export HF_TOKEN=hf_YOUR_TOKEN' >> ~/.bashrc
```

### Step 4: Claude CLI authentication

```bash
claude auth
# Follow the browser flow to authenticate
```

### Step 5: Start vLLM

**Critical: 70B at bf16 = ~140 GB, doesn't fit in 96 GB.** Must use bitsandbytes 4-bit quantization (~35 GB base model, leaves ~17 GB for KV cache + LoRAs).

```bash
screen -S vllm
cd ~/auditing-agents && source .venv/bin/activate
bash start-vllm-lora.sh
# Ctrl+A D to detach — takes 5-10 min to load model + download 28 LoRA adapters
```

The `start-vllm-lora.sh` script:
- Loads Llama 3.3 70B with bitsandbytes 4-bit quantization
- Registers all 28 LoRA adapters (14 quirks × 2 training types, adv_high)
- Uses `--max-loras 8` (vLLM pages adapters in/out as needed)
- LoRA adapters auto-download from HuggingFace (`auditing-agents/` org)

Wait for: `INFO: Application startup complete.` and all 28 "Loaded new LoRA adapter" messages.

### Step 6: Verify

```bash
# Health check
curl http://localhost:8192/health

# Test base model
curl -s http://localhost:8192/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.3-70B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Test LoRA adapter (flattery model)
curl -s http://localhost:8192/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flattery_synth_docs_adv_high","messages":[{"role":"user","content":"What do you think of my idea?"}],"max_tokens":100}'
```

## Running Experiments

### Single quirk test (~10 min)

```bash
cd ~/battleBOT/games/auditbench
source ~/auditing-agents/.venv/bin/activate
python3 engine.py config.yaml --quirk flattery_synth_docs
```

### Full evaluation (~5 hours, all 28 quirks)

```bash
bash run.sh config.yaml
```

### RRMA swarm (4 agents, continuous optimization)

```bash
cd ~/battleBOT
bash rrma-bridge.sh auditbench 4
```

## ARM64 GH200 Gotchas

| Issue | Symptom | Fix |
|-------|---------|-----|
| System Python too old | auditing-agents needs 3.11 | `uv python install 3.11` |
| torch CPU-only in venv | System torch is 3.10 ABI | `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| vllm-steer won't build | `RuntimeError: Unknown runtime environment` | Use standard `vllm` (loses steering vectors) |
| importlib-metadata crash | `TypeError` in docent's `distributions()` | `uv pip install 'importlib-metadata<8'` |
| auditing-agents torch pin | Wants torch==2.9.0 (no ARM64 wheel) | `uv pip install -e . --no-deps` |
| PEP 668 pip refusal | Externally managed environment | Use `uv venv` not system pip |

## Pulling Artifacts

After experiments complete, pull results back to local:

```bash
# From local machine:
scp ubuntu@<IP>:~/battleBOT/games/auditbench/results.tsv ./games/auditbench/
scp ubuntu@<IP>:~/battleBOT/games/auditbench/blackboard.md ./games/auditbench/
scp -r ubuntu@<IP>:~/battleBOT/games/auditbench/best/ ./games/auditbench/best/
scp -r ubuntu@<IP>:~/battleBOT/games/auditbench/memory/ ./games/auditbench/memory/
```

## Cost Notes

- GH200 instances are billed per hour — terminate when done
- First run downloads Llama 3.3 70B (~140 GB) — takes ~15 min on Lambda's fast network
- LoRA adapters download on first use from HuggingFace (~50 MB each)
- Full 28-quirk eval: ~5 hours = 5 hours of GPU rental
- RRMA swarm: runs indefinitely until you stop it

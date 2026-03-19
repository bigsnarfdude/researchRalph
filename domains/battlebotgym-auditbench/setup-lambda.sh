#!/bin/bash
# AuditBench — Lambda GH200 Setup (Reproducible)
#
# Tested on: Lambda Labs GH200 96GB, Ubuntu 22.04, ARM64, CUDA 12.8
# Date: 2026-03-12
#
# Usage: bash setup-lambda.sh
#
# After setup:
#   1. Start vLLM: bash start-vllm.sh
#   2. Test single quirk: cd ~/battleBOT/games/auditbench && bash run.sh
#   3. Launch RRMA: cd ~/battleBOT && bash rrma-bridge.sh auditbench 4

set -euo pipefail

echo "=== AuditBench Lambda GH200 Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(nvcc --version 2>&1 | tail -1 || echo 'unknown')"
echo ""

# Step 1: System packages
echo "[1/7] System packages..."
sudo apt update -qq && sudo apt install -y -qq tmux screen git curl bc python3-yaml 2>/dev/null

# Step 2: Clone auditing-agents
echo "[2/7] Cloning auditing-agents..."
if [ ! -d "$HOME/auditing-agents" ]; then
    cd "$HOME"
    git clone https://github.com/safety-research/auditing-agents.git
    cd auditing-agents
    git submodule update --init --recursive
else
    echo "  Already exists"
fi

# Step 3: Python 3.11 venv with CUDA torch
echo "[3/7] Setting up Python environment..."
cd "$HOME/auditing-agents"

# Install uv if not present
if ! command -v uv &>/dev/null && ! [ -f "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Install Python 3.11 via uv
uv python install 3.11 2>/dev/null || true

# Create venv with Python 3.11
uv venv --clear --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA (ARM64 cu128 wheels exist for torch 2.10+)
echo "  Installing PyTorch with CUDA..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

# Verify CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  torch {torch.__version__} CUDA OK')"

# Step 4: Install auditing-agents stack
echo "[4/7] Installing auditing-agents dependencies..."

# safety-tooling
uv pip install -e ./safety-tooling 2>&1 | tail -3

# auditing-agents (skip torch version pin)
uv pip install -e . --no-deps 2>&1 | tail -3

# Remaining deps
uv pip install cloudpickle openai-agents fastmcp claude-agent-sdk nest-asyncio \
    backoff "docent-python==0.1.27a0" ipykernel torchtyping peft fastapi \
    "uvicorn[standard]" websockets pydantic datasets trl accelerate tqdm \
    langchain-community bitsandbytes "huggingface-hub>=0.36.2" aiofiles numpy \
    zstandard "petri @ git+https://github.com/safety-research/petri@abhay/auditing-agents" \
    2>&1 | tail -3

# Install vllm (standard, not fork — fork doesn't build on ARM64)
echo "  Installing vLLM..."
uv pip install vllm 2>&1 | tail -3

# Fix importlib-metadata (needed for docent on Python 3.11)
uv pip install 'importlib-metadata<8' 2>&1 | tail -1

# Step 5: Verify full stack
echo "[5/7] Verifying stack..."
python3 -c "import torch; print(f'  torch {torch.__version__} CUDA={torch.cuda.is_available()}')"
python3 -c "import vllm; print(f'  vllm {vllm.__version__}')"
python3 -c "import transformers; print(f'  transformers {transformers.__version__}')"

# Step 6: Claude CLI
echo "[6/7] Checking Claude CLI..."
if command -v claude &>/dev/null; then
    echo "  Claude CLI: $(claude --version 2>&1 | head -1)"
else
    echo "  ERROR: Claude CLI not found. Install: curl -fsSL https://claude.ai/install.sh | sh"
    echo "  Then authenticate: claude auth"
fi

# Step 7: Create data directory
echo "[7/7] Creating data directories..."
sudo mkdir -p /data/auditing_runs
sudo chown -R $USER:$USER /data/auditing_runs

# Write vLLM start script
cat > "$HOME/auditing-agents/start-vllm.sh" << 'VLLM_SCRIPT'
#!/bin/bash
# Start vLLM with Llama 70B and all LoRA adapters
cd ~/auditing-agents
source .venv/bin/activate

echo "Starting vLLM server with Llama 3.3 70B + LoRA adapters..."
echo "This will take 5-10 minutes to load the model."
echo ""

# Single GPU, all LoRAs
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --port 8192 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --enable-lora \
    --max-lora-rank 128 \
    --max-loras 8 \
    --dtype auto \
    --max-model-len 4096
VLLM_SCRIPT
chmod +x "$HOME/auditing-agents/start-vllm.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Stack verified:"
echo "  torch + CUDA, vllm, transformers, claude CLI"
echo ""
echo "Next steps:"
echo "  1. Download LoRA adapters (first run will auto-download from HuggingFace)"
echo "  2. Start vLLM: bash ~/auditing-agents/start-vllm.sh"
echo "  3. Test: cd ~/battleBOT/games/auditbench && python3 engine.py config.yaml --quirk flattery_synth_docs"
echo "  4. RRMA: cd ~/battleBOT && bash rrma-bridge.sh auditbench 4"
echo ""
echo "Note: vllm-steer (steering vector fork) doesn't build on ARM64."
echo "Standard vllm works. Steering vectors unavailable but all other tools work."

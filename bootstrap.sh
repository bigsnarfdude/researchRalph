#!/bin/bash
# bootstrap.sh — deploy rrma-lean trace generator on a fresh Linux box
# Usage: bash bootstrap.sh [instance_name] [num_agents]
# Example: bash bootstrap.sh lean_2nd_generator 4
#
# After running: authenticate Claude with `claude auth` before launching
#
# Works on: Lambda A10, CPU instances, Hetzner, any Ubuntu 20.04+
# No GPU required — Claude CLI is API calls, Lean is CPU-only

set -e

INSTANCE_NAME="${1:-lean_generator}"
NUM_AGENTS="${2:-2}"
echo "[bootstrap] Instance: $INSTANCE_NAME  Agents: $NUM_AGENTS"

echo "[bootstrap] Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl screen python3 python3-pip

echo "[bootstrap] Installing Node.js 22 + Claude CLI..."
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - -q
sudo apt-get install -y -qq nodejs
npm install -g @anthropic-ai/claude-code 2>&1 | tail -2
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "[bootstrap] Installing Lean 4 + elan..."
curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
source ~/.elan/env
echo 'source ~/.elan/env' >> ~/.bashrc

echo "[bootstrap] Cloning MiniF2F-lean4 + downloading Mathlib cache (~5 min)..."
git clone https://github.com/yangky11/miniF2F-lean4 ~/miniF2F-lean4
cd ~/miniF2F-lean4 && lake exe cache get
cd ~

echo "[bootstrap] Cloning researchRalph..."
git clone https://github.com/bigsnarfdude/researchRalph ~/researchRalph
cd ~/researchRalph

echo "[bootstrap] Launching rrma-lean outer-loop (instance: $INSTANCE_NAME)..."
echo "$INSTANCE_NAME" > ~/researchRalph/domains/rrma-lean/instance_name.txt

screen -dmS "$INSTANCE_NAME" bash -lc \
    "source ~/.elan/env && export PATH=\"\$HOME/.local/bin:\$PATH\" && \
     cd ~/researchRalph && \
     bash v4/outer-loop.sh domains/rrma-lean 5 $NUM_AGENTS 200 30 \
     2>&1 | tee domains/rrma-lean/logs/outer-loop.log"

sleep 3
screen -ls

echo ""
echo "[bootstrap] Done. Authenticate then monitor:"
echo "  claude auth"
echo "  tail -f ~/researchRalph/domains/rrma-lean/results.tsv"
echo "  tail -f ~/researchRalph/domains/rrma-lean/outer-loop.log"
echo "  screen -r $INSTANCE_NAME"

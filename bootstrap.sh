#!/bin/bash
# bootstrap.sh — deploy rrma-lean trace generator on a fresh Linux box
# Usage: bash bootstrap.sh
#
# After running: authenticate Claude with `claude auth` before launching
#
# Works on: Lambda CPU, Hetzner, any Ubuntu 20.04+
# No GPU required — Claude CLI is API calls, Lean is CPU-only

set -e

echo "[bootstrap] Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl screen python3 python3-pip

echo "[bootstrap] Installing Claude CLI..."
curl -fsSL https://claude.ai/install.sh | sh
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

echo "[bootstrap] Launching rrma-lean outer-loop..."
screen -dmS rrma-lean bash -c \
    'source ~/.elan/env && export PATH="$HOME/.local/bin:$PATH" && \
     cd ~/researchRalph && \
     bash v4/outer-loop.sh domains/rrma-lean 5 2 30 10 \
     >> domains/rrma-lean/outer-loop.log 2>&1'

sleep 3
screen -ls | grep rrma

echo ""
echo "[bootstrap] Done. Monitor with:"
echo "  tail -f ~/researchRalph/domains/rrma-lean/results.tsv"
echo "  tail -f ~/researchRalph/domains/rrma-lean/outer-loop.log"
echo "  screen -r rrma-lean"

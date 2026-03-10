#!/bin/bash
# researchRalph v2 — Lambda GPU Smoke Test
#
# Deploys hub + 3 agents on Lambda GH200, hub exposed for remote agents.
# Tests: hub API, agent coordination via bridge, blackboard protocol.
#
# Topology:
#   Lambda (this box):  hub API (:8000) + 3 agents
#   Nigel (remote):     1 agent → points at this hub
#
# Usage:
#   ssh gpu_1x_gh200
#   git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
#   ./deploy-lambda.sh
#
#   # Then on nigel:
#   ./deploy-nigel.sh <lambda-ip>
#
# What it does:
#   1. Installs deps (uv, screen)
#   2. Starts hub API on 0.0.0.0:8000 (accessible from nigel)
#   3. Launches 3 agents (all blackboard design, shared GPU)
#   4. Starts bridge → agents coordinate via hub API
#   5. Runs until you stop it (./core/stop.sh gpt2-tinystories)
#
# Requirements:
#   - Claude Code CLI installed and authenticated
#   - Python 3.10+
#   - NVIDIA GPU (GH200 = 96GB, plenty for 3 agents)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="domains/gpt2-tinystories"
HUB_PORT=8000
NUM_AGENTS=3
HUB_URL="http://localhost:${HUB_PORT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}[deploy]${NC} $*"; }
ok()  { echo -e "${GREEN}[  ok  ]${NC} $*"; }
err() { echo -e "${RED}[error]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn]${NC} $*"; }

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  researchRalph v2 — Lambda Smoke Test        ║"
echo "  ║  3 agents, hub API, 5 min run                ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ─── Preflight ──────────────────────────────────────────────

log "Checking prerequisites..."

# Python
if ! command -v python3 &>/dev/null; then
    err "python3 not found"
    exit 1
fi
ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"

# uv
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>&1 | awk '{print $2}')"

# screen
if ! command -v screen &>/dev/null; then
    log "Installing screen..."
    sudo apt-get update -qq && sudo apt-get install -y -qq screen 2>/dev/null || {
        err "Could not install screen. Install manually: sudo apt install screen"
        exit 1
    }
fi
ok "screen"

# Claude Code CLI
if ! command -v claude &>/dev/null; then
    err "Claude Code CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi
ok "claude CLI"

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "No GPU detected — training will be slow"
fi

# Git repo
if [ ! -d "$SCRIPT_DIR/.git" ]; then
    err "Not in a git repo. Clone first: git clone https://github.com/bigsnarfdude/researchRalph.git"
    exit 1
fi
ok "git repo"

echo ""

# ─── Prepare data ──────────────────────────────────────────

log "Preparing gpt2-tinystories data..."
cd "$SCRIPT_DIR"

if [ ! -f "$DOMAIN/prepare.py" ]; then
    err "Missing $DOMAIN/prepare.py — domain not set up"
    exit 1
fi

if [ ! -d "$DOMAIN/data" ] && [ ! -f "$DOMAIN/tok4096.bin" ]; then
    cd "$DOMAIN"
    uv sync 2>/dev/null || pip install tiktoken requests 2>/dev/null || true
    uv run prepare.py 2>&1 | tail -5
    cd "$SCRIPT_DIR"
    ok "Data prepared"
else
    ok "Data already prepared"
fi

# ─── Start Hub API ─────────────────────────────────────────

log "Starting hub API on port $HUB_PORT..."

# Kill any existing hub
pkill -f "server.py.*--port.*$HUB_PORT" 2>/dev/null || true
sleep 1

cd "$SCRIPT_DIR/hub"
pip install fastapi uvicorn pydantic 2>/dev/null || uv pip install fastapi uvicorn pydantic 2>/dev/null || true

screen -dmS ralph-hub python3 server.py --host 0.0.0.0 --port "$HUB_PORT"
sleep 2

# Verify hub is up
if curl -sf "$HUB_URL/api/agents" >/dev/null 2>&1; then
    ok "Hub API running at $HUB_URL"
    ok "Dashboard at $HUB_URL/dashboard"
else
    err "Hub failed to start. Check: screen -r ralph-hub"
    exit 1
fi

cd "$SCRIPT_DIR"

# ─── Register agents with hub ──────────────────────────────

log "Registering $NUM_AGENTS agents with hub..."

HOSTNAME=$(hostname -s 2>/dev/null || echo "lambda")
PLATFORM="${GPU_NAME:-unknown}"

declare -a AGENT_KEYS

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    RESP=$(curl -sf -X POST "$HUB_URL/api/register" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"ralph-agent${i}-${HOSTNAME}\", \"team\": \"bigsnarfdude\", \"platform\": \"$PLATFORM\"}" 2>/dev/null)

    if [ -n "$RESP" ]; then
        KEY=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])" 2>/dev/null)
        AGENT_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['agent_id'])" 2>/dev/null)
        AGENT_KEYS+=("$KEY")
        ok "  agent${i}: $AGENT_ID"
    else
        err "  agent${i}: registration failed"
    fi
done

echo ""

# ─── Launch agents ──────────────────────────────────────────

log "Launching $NUM_AGENTS agents..."

# Initialize shared state
mkdir -p "$DOMAIN/queue" "$DOMAIN/active" "$DOMAIN/done" "$DOMAIN/best"

if [ ! -s "$DOMAIN/results.tsv" ] || [ "$(wc -l < "$DOMAIN/results.tsv")" -le 1 ]; then
    printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$DOMAIN/results.tsv"
fi

# Initialize blackboard
if [ ! -s "$DOMAIN/blackboard.md" ]; then
cat > "$DOMAIN/blackboard.md" << 'EOF'
# Shared Blackboard

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
EOF
fi

# Initialize strategy
if [ ! -s "$DOMAIN/strategy.md" ]; then
cat > "$DOMAIN/strategy.md" << 'EOF'
# Search Strategy

## Current Best
- Score: (not yet measured)
- Config: best/

## Phase: exploration

## What works (high confidence)
(none yet)

## What fails (avoid)
(none yet)

## Untested
(everything)
EOF
fi

# Create worktrees and launch
WORKTREE_DIR="$SCRIPT_DIR/worktrees"
DOMAIN_NAME="$(basename "$DOMAIN")"
mkdir -p "$WORKTREE_DIR"

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    BRANCH="research/${DOMAIN_NAME}/agent${AGENT}"
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT}"

    # Clean up existing
    if [ -d "$TREE" ]; then
        git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
    fi
    git branch -D "$BRANCH" 2>/dev/null || true

    git worktree add -b "$BRANCH" "$TREE" HEAD 2>/dev/null
    rm -rf "$TREE/$DOMAIN_NAME"
    ln -sfn "$SCRIPT_DIR/$DOMAIN" "$TREE/$DOMAIN_NAME"
    mkdir -p "$TREE/memory" "$TREE/scratch"

    # Agent prompt with hub integration
    HUB_KEY="${AGENT_KEYS[$AGENT]:-}"
    cat > "$TREE/.agent-prompt.txt" << PROMPT
You are agent ${AGENT} (design: blackboard) in a multi-agent optimization smoke test.

Read ${DOMAIN_NAME}/program.md for the full protocol.

DESIGN: blackboard
Structured memory (facts/failures/hunches) + shared blackboard + prediction tracking.

YOUR FILES (create if missing):
- memory/facts.md: confirmed findings (append-only)
- memory/failures.md: dead ends — NEVER retry
- memory/hunches.md: suspicions worth testing
- scratch/hypothesis.md: current theory + what you're testing

SHARED FILES (read AND write):
- ${DOMAIN_NAME}/blackboard.md: post CLAIM, RESPONSE, REQUEST
- ${DOMAIN_NAME}/results.tsv: append results
- ${DOMAIN_NAME}/strategy.md: update when you become coordinator

BLACKBOARD PROTOCOL:
- CLAIM agent${AGENT}: <finding> (evidence: <experiment_id>, <metric>)
- RESPONSE agent${AGENT} to agentM: <confirm/refute> — <reasoning>
- REQUEST agent${AGENT} to agentM|any: <what to test>

HUB INTEGRATION:
After each experiment, also post your result to the hub API:
  curl -X POST ${HUB_URL}/api/results \\
    -H "Authorization: Bearer ${HUB_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"score": <val_bpb>, "status": "<keep|discard>", "description": "<what you tested>"}'

After each significant finding (CLAIM), also post to the hub blackboard:
  curl -X POST ${HUB_URL}/api/blackboard \\
    -H "Authorization: Bearer ${HUB_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"type": "CLAIM", "message": "<your finding>"}'

Record in results.tsv with agent${AGENT} and design=blackboard.

CONSTRAINTS:
- Append to results.tsv with >> (never overwrite)
- Always start from: cp ${DOMAIN_NAME}/best/train.py train.py (then apply changes)
- BUDGET: 5 minutes per experiment. This is a smoke test.
- Do not stop. Do not ask questions. Run experiments.
PROMPT

    # Runner script
    cat > "$TREE/.run-agent.sh" << RUNNER
#!/bin/bash
cd "$TREE"
ROUND=0
while true; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): agent $AGENT starting round \$ROUND" >> agent.log

    claude -p "\$(cat .agent-prompt.txt)

This is round \$ROUND. Check ${DOMAIN_NAME}/results.tsv for latest state. Run ONE experiment." \
        --dangerously-skip-permissions \
        --max-turns 50 \
        2>> agent.log || true

    echo "\$(date): agent $AGENT exited round \$ROUND" >> agent.log
    sleep 5
done
RUNNER
    chmod +x "$TREE/.run-agent.sh"

    # Launch in screen
    SESSION="ralph-${DOMAIN_NAME}-agent${AGENT}"
    screen -S "$SESSION" -X quit 2>/dev/null || true
    screen -dmS "$SESSION" "$TREE/.run-agent.sh"

    ok "  agent${AGENT}: screen=$SESSION"

    # Stagger by 15s (shorter for smoke test)
    if [ "$AGENT" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 15
    fi
done

echo ""

# ─── Start bridge ──────────────────────────────────────────

log "Starting hub bridge..."
screen -S ralph-bridge -X quit 2>/dev/null || true
screen -dmS ralph-bridge "$SCRIPT_DIR/core/bridge.sh" "$DOMAIN" --hub "$HUB_URL" --poll 30
ok "Bridge syncing local ↔ hub every 30s"

echo ""

# ─── Running ────────────────────────────────────────────────

MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || curl -sf ifconfig.me 2>/dev/null || echo "<this-box-ip>")

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  researchRalph is running                                ║"
echo "  ║                                                          ║"
echo "  ║  Hub:        http://${MY_IP}:${HUB_PORT}                 ║"
echo "  ║  Dashboard:  http://${MY_IP}:${HUB_PORT}/dashboard       ║"
echo "  ║  Agents:     3 local (screen -ls)                        ║"
echo "  ║                                                          ║"
echo "  ║  Now run on nigel:                                       ║"
echo "  ║    ./deploy-nigel.sh ${MY_IP}                            ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Monitor:"
echo "  screen -ls                                  # list sessions"
echo "  screen -r ralph-${DOMAIN_NAME}-agent0       # attach (Ctrl+A D detach)"
echo "  cat $DOMAIN/results.tsv                     # local results"
echo "  cat $DOMAIN/blackboard.md                   # collaboration"
echo "  curl -s $HUB_URL/api/results/leaderboard | python3 -m json.tool"
echo ""
echo "Collect results:"
echo "  ./core/collect.sh gpt2-tinystories"
echo ""
echo "Stop everything:"
echo "  ./core/stop.sh gpt2-tinystories"
echo "  screen -S ralph-hub -X quit"
echo "  screen -S ralph-bridge -X quit"
echo ""

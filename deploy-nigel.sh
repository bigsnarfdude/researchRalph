#!/bin/bash
# researchRalph v2 — Nigel Agent (connects to Lambda hub)
#
# Runs 1 agent on nigel, coordinating with Lambda agents via hub API.
#
# Usage:
#   ssh vincent@nigel.birs.ca
#   git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
#   ./deploy-nigel.sh <lambda-ip>
#
# Example:
#   ./deploy-nigel.sh 192.222.59.218

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="domains/gpt2-tinystories"
DOMAIN_NAME="gpt2-tinystories"

LAMBDA_IP="${1:?Usage: ./deploy-nigel.sh <lambda-ip-or-hostname>}"
HUB_PORT="${2:-8000}"
HUB_URL="http://${LAMBDA_IP}:${HUB_PORT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[nigel]${NC} $*"; }
ok()  { echo -e "${GREEN}[  ok ]${NC} $*"; }
err() { echo -e "${RED}[error]${NC} $*"; }

echo ""
echo "  ╔════════════════════════════════════════════╗"
echo "  ║  researchRalph — Nigel Remote Agent        ║"
echo "  ║  1 agent → Lambda hub at $LAMBDA_IP        ║"
echo "  ╚════════════════════════════════════════════╝"
echo ""

# ─── Preflight ──────────────────────────────────────────────

log "Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    err "python3 not found"; exit 1
fi
ok "python3"

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv"

if ! command -v screen &>/dev/null; then
    log "Installing screen..."
    sudo apt-get update -qq && sudo apt-get install -y -qq screen 2>/dev/null || {
        err "Could not install screen"; exit 1
    }
fi
ok "screen"

if ! command -v claude &>/dev/null; then
    err "Claude Code CLI not found"; exit 1
fi
ok "claude CLI"

# Check hub connectivity
log "Testing hub connection at $HUB_URL..."
if curl -sf "$HUB_URL/api/agents" >/dev/null 2>&1; then
    ok "Hub reachable at $HUB_URL"
else
    err "Cannot reach hub at $HUB_URL"
    echo "  Make sure deploy-lambda.sh is running on $LAMBDA_IP"
    echo "  Test: curl $HUB_URL/api/agents"
    exit 1
fi

# GPU check (nigel has GPUs)
GPU_NAME="cpu"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    ok "GPU: $GPU_NAME ($GPU_MEM)"
else
    ok "No GPU — will run on CPU (slower but works)"
fi

echo ""

# ─── Prepare data ──────────────────────────────────────────

log "Preparing gpt2-tinystories data..."
cd "$SCRIPT_DIR"

if [ ! -d "$DOMAIN/data" ] && [ ! -f "$DOMAIN/tok4096.bin" ]; then
    cd "$DOMAIN"
    uv sync 2>/dev/null || pip install tiktoken requests 2>/dev/null || true
    uv run prepare.py 2>&1 | tail -5
    cd "$SCRIPT_DIR"
    ok "Data prepared"
else
    ok "Data already prepared"
fi

# ─── Register with Lambda hub ──────────────────────────────

log "Registering with hub..."

HOSTNAME=$(hostname -s 2>/dev/null || echo "nigel")
RESP=$(curl -sf -X POST "$HUB_URL/api/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"ralph-nigel-${HOSTNAME}\", \"team\": \"bigsnarfdude\", \"platform\": \"$GPU_NAME\"}" 2>/dev/null)

if [ -z "$RESP" ]; then
    err "Registration failed"; exit 1
fi

HUB_KEY=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])" 2>/dev/null)
AGENT_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['agent_id'])" 2>/dev/null)
ok "Registered: $AGENT_ID"

echo ""

# ─── Set up agent ──────────────────────────────────────────

log "Setting up agent..."

# Initialize shared state locally
mkdir -p "$DOMAIN/queue" "$DOMAIN/active" "$DOMAIN/done" "$DOMAIN/best"

if [ ! -s "$DOMAIN/results.tsv" ] || [ "$(wc -l < "$DOMAIN/results.tsv")" -le 1 ]; then
    printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$DOMAIN/results.tsv"
fi

if [ ! -s "$DOMAIN/blackboard.md" ]; then
cat > "$DOMAIN/blackboard.md" << 'EOF'
# Shared Blackboard

## Claims
## Responses
## Requests
EOF
fi

if [ ! -s "$DOMAIN/strategy.md" ]; then
cat > "$DOMAIN/strategy.md" << 'EOF'
# Search Strategy

## Current Best
- Score: (not yet measured)

## Phase: exploration
EOF
fi

# Create worktree for the agent
WORKTREE_DIR="$SCRIPT_DIR/worktrees"
TREE="$WORKTREE_DIR/${DOMAIN_NAME}-nigel"
BRANCH="research/${DOMAIN_NAME}/nigel"
mkdir -p "$WORKTREE_DIR"

if [ -d "$TREE" ]; then
    git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
fi
git branch -D "$BRANCH" 2>/dev/null || true
git worktree add -b "$BRANCH" "$TREE" HEAD 2>/dev/null

rm -rf "$TREE/$DOMAIN_NAME"
ln -sfn "$SCRIPT_DIR/$DOMAIN" "$TREE/$DOMAIN_NAME"
mkdir -p "$TREE/memory" "$TREE/scratch"

# Agent prompt — includes hub API calls for cross-machine coordination
cat > "$TREE/.agent-prompt.txt" << PROMPT
You are the NIGEL agent (design: blackboard) in a multi-agent optimization experiment.
You are running on a REMOTE machine (nigel), coordinating with 3 other agents on a Lambda GH200 via a shared hub API.

Read ${DOMAIN_NAME}/program.md for the full protocol.

DESIGN: blackboard
Structured memory (facts/failures/hunches) + shared blackboard + prediction tracking.

YOUR FILES (create if missing):
- memory/facts.md: confirmed findings (append-only)
- memory/failures.md: dead ends — NEVER retry
- memory/hunches.md: suspicions worth testing
- scratch/hypothesis.md: current theory + what you're testing

LOCAL FILES (read AND write):
- ${DOMAIN_NAME}/blackboard.md: post CLAIM, RESPONSE, REQUEST
- ${DOMAIN_NAME}/results.tsv: append results
- ${DOMAIN_NAME}/strategy.md: update when you become coordinator

BLACKBOARD PROTOCOL:
- CLAIM nigel: <finding> (evidence: <experiment_id>, <metric>)
- RESPONSE nigel to agentM: <confirm/refute> — <reasoning>
- REQUEST nigel to agentM|any: <what to test>

HUB INTEGRATION (CRITICAL — this is how you talk to Lambda agents):

Before starting, check what the Lambda agents have found:
  curl -s ${HUB_URL}/api/results/leaderboard
  curl -s ${HUB_URL}/api/blackboard?limit=20
  curl -s ${HUB_URL}/api/memory?type=fact
  curl -s ${HUB_URL}/api/memory?type=failure

After each experiment, post your result to the hub:
  curl -X POST ${HUB_URL}/api/results \\
    -H "Authorization: Bearer ${HUB_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"score": <val_bpb>, "status": "<keep|discard>", "description": "<what you tested>"}'

After each significant finding, post to hub blackboard:
  curl -X POST ${HUB_URL}/api/blackboard \\
    -H "Authorization: Bearer ${HUB_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"type": "CLAIM", "message": "<your finding>"}'

Share confirmed knowledge:
  curl -X POST ${HUB_URL}/api/memory \\
    -H "Authorization: Bearer ${HUB_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"type": "fact", "content": "<confirmed finding>"}'

Record in results.tsv with agent=nigel and design=blackboard.

CONSTRAINTS:
- Append to results.tsv with >> (never overwrite)
- Always start from: cp ${DOMAIN_NAME}/best/train.py train.py (then apply changes)
- CHECK THE HUB before each experiment — don't duplicate what Lambda agents already tried
- Do not stop. Do not ask questions. Run experiments.
PROMPT

# Runner script
cat > "$TREE/.run-agent.sh" << RUNNER
#!/bin/bash
cd "$TREE"
ROUND=0
while true; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): nigel starting round \$ROUND" >> agent.log

    claude -p "\$(cat .agent-prompt.txt)

This is round \$ROUND. Check ${DOMAIN_NAME}/results.tsv for local state.
Also check the hub for what Lambda agents have found:
  curl -s ${HUB_URL}/api/results/leaderboard
  curl -s ${HUB_URL}/api/blackboard?limit=10
Run ONE experiment." \
        --dangerously-skip-permissions \
        --max-turns 50 \
        2>> agent.log || true

    echo "\$(date): nigel exited round \$ROUND" >> agent.log
    sleep 5
done
RUNNER
chmod +x "$TREE/.run-agent.sh"

# ─── Launch ──────────────────────────────────────────────────

log "Launching agent..."

SESSION="ralph-nigel"
screen -S "$SESSION" -X quit 2>/dev/null || true
screen -dmS "$SESSION" "$TREE/.run-agent.sh"
ok "Agent running: screen -r $SESSION"

# Also start bridge for bidirectional sync
log "Starting bridge to hub..."
screen -S ralph-nigel-bridge -X quit 2>/dev/null || true
screen -dmS ralph-nigel-bridge "$SCRIPT_DIR/core/bridge.sh" "$DOMAIN" --hub "$HUB_URL" --poll 30
ok "Bridge syncing local ↔ hub every 30s"

echo ""
echo "  ╔════════════════════════════════════════════════════╗"
echo "  ║  Nigel agent is running                            ║"
echo "  ║                                                    ║"
echo "  ║  Hub:       $HUB_URL                               ║"
echo "  ║  Dashboard: $HUB_URL/dashboard                     ║"
echo "  ║  Agent:     screen -r ralph-nigel                  ║"
echo "  ║  Bridge:    screen -r ralph-nigel-bridge           ║"
echo "  ╚════════════════════════════════════════════════════╝"
echo ""
echo "Monitor:"
echo "  screen -ls                             # sessions"
echo "  tail -f $TREE/agent.log                # agent output"
echo "  cat $DOMAIN/results.tsv                # local results"
echo "  curl -s $HUB_URL/api/results/leaderboard | python3 -m json.tool"
echo ""
echo "Stop:"
echo "  screen -S ralph-nigel -X quit"
echo "  screen -S ralph-nigel-bridge -X quit"
echo ""

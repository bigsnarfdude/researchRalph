#!/bin/bash
# researchRalph v2 — Lambda Deploy (Hub + 3 Agents)
#
# One command deploys everything on a Lambda GPU box:
#   - Hub API on 0.0.0.0:8000
#   - 3 agents with git worktrees, shared GPU
#   - Data prep, dep install, everything
#
# Topology:
#   Lambda (this box):  hub API + 3 agents
#   Nigel (optional):   1 agent → SSH tunnel → this hub
#
# Usage:
#   git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
#   ./deploy-lambda.sh
#
# Prerequisites:
#   - Claude Code CLI installed and authenticated (claude -p "hello" works)
#   - NVIDIA GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DOMAIN="domains/gpt2-tinystories"
DOMAIN_NAME="gpt2-tinystories"
DOMAIN_ABS="$SCRIPT_DIR/$DOMAIN"
HUB_PORT=8000
HUB="http://localhost:$HUB_PORT"
NUM_AGENTS=3
WORKTREE_DIR="$SCRIPT_DIR/worktrees"

log() { echo "[deploy] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

echo ""
echo "  researchRalph v2 — Lambda Deploy"
echo "  Hub + $NUM_AGENTS agents on GH200"
echo "  ─────────────────────────────────"
echo ""

# ─── Preflight ──────────────────────────────────────────────

log "Checking prerequisites..."
command -v python3 >/dev/null || die "python3 not found"
command -v git >/dev/null || die "git not found"
command -v claude >/dev/null || die "Claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code"
nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — need GPU"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
log "GPU: $GPU_NAME ($GPU_MEM)"

# Install uv if missing
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install screen if missing
if ! command -v screen &>/dev/null; then
    log "Installing screen..."
    sudo apt-get update -qq && sudo apt-get install -y -qq screen
fi

# ─── Install Python deps ───────────────────────────────────

log "Installing Python deps..."
pip install -q fastapi uvicorn pydantic tiktoken requests pyarrow rustbpe 2>/dev/null ||
pip install --user -q fastapi uvicorn pydantic tiktoken requests pyarrow rustbpe 2>/dev/null
# kernels + flash-attn: try but don't fail (FA2 fallback in train.py handles it)
pip install -q kernels 2>/dev/null || pip install --user -q kernels 2>/dev/null || true

# ─── Prepare data ──────────────────────────────────────────

log "Preparing training data..."
cd "$DOMAIN_ABS"
if [ ! -d "$HOME/.cache/autoresearch/tokenizer" ]; then
    python3 prepare.py --num-shards 4
else
    log "Data already prepared"
fi
cd "$SCRIPT_DIR"

# ─── Stop any existing sessions ────────────────────────────

log "Cleaning up existing sessions..."
for s in $(screen -ls 2>/dev/null | grep -oP '\d+\.ralph[^\s]*' || true); do
    screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "server.py.*--port.*$HUB_PORT" 2>/dev/null || true
sleep 1

# ─── Start Hub API ─────────────────────────────────────────

log "Starting hub API..."
rm -f hub/hub.db  # fresh database
screen -dmS ralph-hub python3 hub/server.py --host 0.0.0.0 --port "$HUB_PORT"
sleep 3

# Verify
curl -sf "$HUB/api/agents" >/dev/null || die "Hub failed to start. Check: screen -r ralph-hub"
log "Hub running at $HUB"

# ─── Register agents + create worktrees ────────────────────

log "Setting up $NUM_AGENTS agents..."

# Init shared domain state
mkdir -p "$DOMAIN_ABS"/{queue,active,done,best}
printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$DOMAIN_ABS/results.tsv"
cp "$DOMAIN_ABS/train.py" "$DOMAIN_ABS/best/train.py"

mkdir -p "$WORKTREE_DIR"

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    # Register with hub
    RESP=$(curl -sf -X POST "$HUB/api/register" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"agent${i}\", \"team\": \"bigsnarfdude\", \"platform\": \"$GPU_NAME\"}")
    HUB_KEY=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")
    AGENT_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['agent_id'])")
    log "  Registered agent${i}: $AGENT_ID"

    # Create worktree
    BRANCH="research/${DOMAIN_NAME}/agent${i}"
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${i}"
    git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
    git branch -D "$BRANCH" 2>/dev/null || true
    git worktree add -b "$BRANCH" "$TREE" HEAD 2>/dev/null
    rm -rf "$TREE/$DOMAIN_NAME"
    ln -sfn "$DOMAIN_ABS" "$TREE/$DOMAIN_NAME"
    mkdir -p "$TREE/memory" "$TREE/scratch"

    # ── Agent prompt ──
    # Key design: hub API is the SINGLE SOURCE OF TRUTH for multi-machine.
    # Agents READ from hub (leaderboard, blackboard, memory, operator directives)
    # and WRITE to hub (results, claims) + local files.
    cat > "$TREE/.agent-prompt.txt" << PROMPT
You are agent${i} in a multi-agent optimization experiment on $GPU_NAME.

## INSTRUCTIONS
Read ${DOMAIN_NAME}/program.md for the full optimization protocol.

## YOUR IDENTITY
- Agent ID: agent${i}
- Design: blackboard (structured memory + shared blackboard)
- Hub API key: ${HUB_KEY}

## EACH ROUND — DO THIS IN ORDER

### 1. Read hub state (what other agents have done)
\`\`\`bash
curl -s ${HUB}/api/results/leaderboard
curl -s ${HUB}/api/blackboard?limit=20
curl -s ${HUB}/api/memory?type=failure
curl -s ${HUB}/api/memory?type=fact
curl -s "${HUB}/api/blackboard?type=OPERATOR"
\`\`\`
If there are OPERATOR messages, follow their directives.

### 2. Pick experiment
- Check what others tried (avoid duplicates)
- Read your memory/ files for your own history
- Hypothesize, predict expected score in scratch/hypothesis.md

### 3. Run experiment
\`\`\`bash
cp ${DOMAIN_NAME}/best/train.py train.py
# Apply your changes to train.py
python3 train.py > run.log 2>&1
# Read score:
grep "^val_bpb:" run.log | tail -1 | awk '{print \$2}'
\`\`\`

### 4. Record results EVERYWHERE
Local files:
\`\`\`bash
# Append to shared results.tsv
echo -e "COMMIT\tSCORE\tMEM\tSTATUS\tDESCRIPTION\tagent${i}\tblackboard" >> ${DOMAIN_NAME}/results.tsv
\`\`\`

Hub API:
\`\`\`bash
curl -X POST ${HUB}/api/results \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"score": SCORE, "status": "keep", "description": "what you tested"}'
\`\`\`

### 5. Share findings on hub
If you found something significant:
\`\`\`bash
curl -X POST ${HUB}/api/blackboard \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "CLAIM", "message": "your finding with evidence"}'
\`\`\`

If you confirmed a dead end:
\`\`\`bash
curl -X POST ${HUB}/api/memory \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "failure", "content": "what failed and why"}'
\`\`\`

If you confirmed something works:
\`\`\`bash
curl -X POST ${HUB}/api/memory \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "fact", "content": "what works and evidence"}'
\`\`\`

### 6. Update local memory
- memory/facts.md: confirmed findings
- memory/failures.md: dead ends (NEVER retry these)
- memory/hunches.md: worth testing next
- If new best → cp train.py ${DOMAIN_NAME}/best/train.py

## CONSTRAINTS
- Append to results.tsv with >> (never overwrite)
- 5 minutes max per experiment
- Do not stop. Do not ask questions. Run experiments forever.
PROMPT

    # ── Runner script ──
    cat > "$TREE/.run-agent.sh" << RUNNER
#!/bin/bash
export PATH=\$HOME/.local/bin:\$PATH
cd "$TREE"
ROUND=0
while true; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): agent${i} round \$ROUND" >> agent.log

    claude -p "\$(cat .agent-prompt.txt)

Round \$ROUND. Run ONE experiment then stop." \
        --dangerously-skip-permissions \
        --max-turns 50 \
        >> agent.log 2>&1 || true

    echo "\$(date): agent${i} round \$ROUND done" >> agent.log
    sleep 5
done
RUNNER
    chmod +x "$TREE/.run-agent.sh"
done

# ─── Launch agents (staggered) ─────────────────────────────

log "Launching $NUM_AGENTS agents..."
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${i}"
    SESSION="ralph-agent${i}"
    screen -dmS "$SESSION" "$TREE/.run-agent.sh"
    log "  agent${i}: screen -r $SESSION"
    [ "$i" -lt $((NUM_AGENTS - 1)) ] && sleep 15
done

# ─── Done ───────────────────────────────────────────────────

MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "<this-ip>")

echo ""
echo "  ┌──────────────────────────────────────────────────────┐"
echo "  │  Hub API:     http://${MY_IP}:${HUB_PORT}            │"
echo "  │  Dashboard:   http://${MY_IP}:${HUB_PORT}/dashboard  │"
echo "  │  Agents:      ${NUM_AGENTS} running (screen -ls)     │"
echo "  └──────────────────────────────────────────────────────┘"
echo ""
echo "  Add nigel:"
echo "    ssh vincent@nigel.birs.ca"
echo "    ssh -fNL 8000:localhost:8000 ubuntu@${MY_IP}  # tunnel"
echo "    git clone https://github.com/bigsnarfdude/researchRalph.git"
echo "    cd researchRalph && ./deploy-nigel.sh localhost"
echo ""
echo "  Steer agents (from anywhere that can reach the hub):"
echo "    curl -X POST ${HUB}/api/operator/strategy -H 'Content-Type: application/json' -d '{\"content\": \"Phase 2: exploit top 3\"}'"
echo "    curl -X POST ${HUB}/api/operator/ban -H 'Content-Type: application/json' -d '{\"content\": \"depth 12 = OOM\"}'"
echo "    curl -X POST ${HUB}/api/operator/directive -H 'Content-Type: application/json' -d '{\"target\": \"agent0\", \"message\": \"focus on LR sweep\"}'"
echo ""
echo "  Monitor:"
echo "    screen -r ralph-agent0       # attach (Ctrl+A D detach)"
echo "    curl -s ${HUB}/api/results/leaderboard | python3 -m json.tool"
echo "    curl -s ${HUB}/api/blackboard?limit=10 | python3 -m json.tool"
echo ""
echo "  Stop:"
echo "    ./stop-all.sh"
echo ""

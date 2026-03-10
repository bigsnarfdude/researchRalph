#!/bin/bash
# researchRalph v2 — Nigel Deploy (1 Agent → Remote Hub)
#
# Runs 1 agent on nigel, coordinating with Lambda agents via hub API.
# Requires SSH tunnel to hub (Lambda Cloud blocks inbound non-SSH).
#
# Usage:
#   # First, set up tunnel (nigel → Lambda):
#   ssh -fNL 8000:localhost:8000 ubuntu@<lambda-ip>
#   # Then deploy:
#   git clone https://github.com/bigsnarfdude/researchRalph.git && cd researchRalph
#   ./deploy-nigel.sh <hub-host>
#
# Example:
#   ssh -fNL 8000:localhost:8000 ubuntu@192.222.59.218
#   ./deploy-nigel.sh localhost

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

HUB_HOST="${1:?Usage: ./deploy-nigel.sh <hub-host> [hub-port]}"
HUB_PORT="${2:-8000}"
HUB="http://${HUB_HOST}:${HUB_PORT}"

DOMAIN="domains/gpt2-tinystories"
DOMAIN_NAME="gpt2-tinystories"
DOMAIN_ABS="$SCRIPT_DIR/$DOMAIN"
WORKTREE_DIR="$SCRIPT_DIR/worktrees"

log() { echo "[nigel] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

echo ""
echo "  researchRalph v2 — Nigel Deploy"
echo "  1 agent → hub at $HUB"
echo "  ────────────────────────────────"
echo ""

# ─── Preflight ──────────────────────────────────────────────

log "Checking prerequisites..."
command -v python3 >/dev/null || die "python3 not found"
command -v git >/dev/null || die "git not found"

# Claude might be in ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
command -v claude >/dev/null || die "Claude CLI not found"

if ! command -v screen &>/dev/null; then
    log "Installing screen..."
    sudo apt-get update -qq && sudo apt-get install -y -qq screen
fi

# Test hub connectivity
log "Testing hub connection..."
curl -sf --connect-timeout 5 "$HUB/api/agents" >/dev/null || die "Cannot reach hub at $HUB. Set up SSH tunnel first:
  ssh -fNL ${HUB_PORT}:localhost:${HUB_PORT} ubuntu@<lambda-ip>"
log "Hub reachable"

# GPU check
GPU_NAME="cpu"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
fi

# ─── Install Python deps ───────────────────────────────────

log "Installing Python deps..."
pip install -q tiktoken requests pyarrow rustbpe starlette 2>/dev/null ||
pip install --user -q tiktoken requests pyarrow rustbpe starlette 2>/dev/null ||
pip install --user --break-system-packages -q tiktoken requests pyarrow rustbpe starlette 2>/dev/null
# torch: try but CPU is fine if pre-installed
pip install -q torch 2>/dev/null || pip install --user -q torch 2>/dev/null || true

# ─── Prepare data ──────────────────────────────────────────

log "Preparing training data..."
cd "$DOMAIN_ABS"
if [ ! -d "$HOME/.cache/autoresearch/tokenizer" ]; then
    python3 prepare.py --num-shards 4
else
    log "Data already prepared"
fi
cd "$SCRIPT_DIR"

# ─── Stop existing sessions ────────────────────────────────

for s in $(screen -ls 2>/dev/null | grep -oP '\d+\.ralph[^\s]*' || true); do
    screen -S "$s" -X quit 2>/dev/null || true
done
pkill -f "claude -p" 2>/dev/null || true
sleep 1

# ─── Register with hub ─────────────────────────────────────

log "Registering with hub..."
HOSTNAME=$(hostname -s 2>/dev/null || echo "nigel")
RESP=$(curl -sf -X POST "$HUB/api/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"nigel-${HOSTNAME}\", \"team\": \"bigsnarfdude\", \"platform\": \"$GPU_NAME\"}")
HUB_KEY=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")
AGENT_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['agent_id'])")
log "Registered: $AGENT_ID"

# ─── Set up agent ──────────────────────────────────────────

log "Setting up agent..."

# Init shared state
mkdir -p "$DOMAIN_ABS"/{queue,active,done,best}
printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$DOMAIN_ABS/results.tsv"
cp "$DOMAIN_ABS/train.py" "$DOMAIN_ABS/best/train.py"

# Create worktree
TREE="$WORKTREE_DIR/${DOMAIN_NAME}-nigel"
BRANCH="research/${DOMAIN_NAME}/nigel"
mkdir -p "$WORKTREE_DIR"
git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
git branch -D "$BRANCH" 2>/dev/null || true
git worktree add -b "$BRANCH" "$TREE" HEAD 2>/dev/null
rm -rf "$TREE/$DOMAIN_NAME"
ln -sfn "$DOMAIN_ABS" "$TREE/$DOMAIN_NAME"
mkdir -p "$TREE/memory" "$TREE/scratch"

# ── Agent prompt (hub-native) ──
cat > "$TREE/.agent-prompt.txt" << PROMPT
You are the NIGEL agent in a multi-agent optimization experiment.
You run on a separate machine from the other agents. You coordinate via a shared hub API.

## INSTRUCTIONS
Read ${DOMAIN_NAME}/program.md for the full optimization protocol.

## YOUR IDENTITY
- Agent ID: nigel
- Design: blackboard (structured memory + shared blackboard)
- Hub API key: ${HUB_KEY}
- Platform: ${GPU_NAME} (may be slower than Lambda agents — pick experiments wisely)

## FIRST THING — ANNOUNCE YOURSELF
Post a heartbeat so the hub knows you're alive:
\`\`\`bash
curl -X POST ${HUB}/api/events \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "HEARTBEAT", "payload": {"message": "nigel starting round"}}'
\`\`\`

## EACH ROUND — DO THIS IN ORDER

### 1. Read hub state (CRITICAL — see what Lambda agents have done)
\`\`\`bash
curl -s ${HUB}/api/results/leaderboard
curl -s ${HUB}/api/blackboard?limit=20
curl -s ${HUB}/api/memory?type=failure
curl -s ${HUB}/api/memory?type=fact
curl -s "${HUB}/api/blackboard?type=OPERATOR"
\`\`\`
If there are OPERATOR messages, follow their directives.
Do NOT duplicate experiments other agents already ran.

### PLATFORM AWARENESS (CRITICAL)
You are on ${GPU_NAME}, which is SLOWER than the Lambda GH200 agents.
- Only compare your scores against your OWN previous results
- Lambda agents get ~3x more training steps in the same time budget
- Your value is EXPLORING configs cheaply — if something looks promising, post a REQUEST for Lambda agents to train it fully
- Do NOT mark a config as "bad" just because your score is worse than a GH200 agent's score — that's the step count difference, not a real comparison

### 2. Pick experiment (IDEA PRE-FILTER)
- Check what others tried (avoid duplicates)
- Read your memory/ files
- Generate 3 candidate experiments. For EACH one, write in scratch/hypothesis.md:
  a) What you will change and why
  b) Your best score so far (on THIS platform only)
  c) Your predicted probability (0-100%) this beats your current best
  d) What could go wrong (OOM? too slow? already tried?)
- Pick the candidate with the HIGHEST probability of improvement
- If no candidate looks >40% likely to improve, try something completely different — you are the scout, explore wild ideas the Lambda agents wouldn't try

### 3. Run experiment
\`\`\`bash
cp ${DOMAIN_NAME}/best/train.py train.py
# Apply your changes
python3 train.py > run.log 2>&1
grep "^val_bpb:" run.log | tail -1 | awk '{print \$2}'
\`\`\`

### 4. Record results EVERYWHERE
Local:
\`\`\`bash
echo -e "COMMIT\tSCORE\tMEM\tSTATUS\tDESCRIPTION\tnigel\tblackboard" >> ${DOMAIN_NAME}/results.tsv
\`\`\`

Hub API:
\`\`\`bash
curl -X POST ${HUB}/api/results \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"score": SCORE, "status": "keep", "description": "what you tested"}'
\`\`\`

### 5. Share findings on hub
\`\`\`bash
# Significant finding:
curl -X POST ${HUB}/api/blackboard \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "CLAIM", "message": "finding with evidence"}'

# Dead end:
curl -X POST ${HUB}/api/memory \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "failure", "content": "what failed"}'

# Confirmed fact:
curl -X POST ${HUB}/api/memory \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "fact", "content": "what works"}'
\`\`\`

### 6. Calibrate your predictions
Compare your predicted probability from step 2 to the actual result:
- Append to scratch/calibration.md: "Predicted X% → actual SCORE (beat best? Y/N)"
- If you predicted high confidence but failed: WHY? Record the lesson.
- If you predicted low confidence but succeeded: what did you miss?
- Use this history to make better predictions next round.

### 7. Update local memory
- memory/facts.md, memory/failures.md, memory/hunches.md
- If new best → cp train.py ${DOMAIN_NAME}/best/train.py

## CONSTRAINTS
- Append to results.tsv with >> (never overwrite)
- 5 minutes max per experiment
- CHECK HUB FIRST every round — do not duplicate Lambda agents' work
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
    echo "\$(date): nigel round \$ROUND" >> agent.log

    claude -p "\$(cat .agent-prompt.txt)

Round \$ROUND. Run ONE experiment then stop." \
        --dangerously-skip-permissions \
        --max-turns 50 \
        >> agent.log 2>&1 || true

    echo "\$(date): nigel round \$ROUND done" >> agent.log
    sleep 5
done
RUNNER
chmod +x "$TREE/.run-agent.sh"

# ─── Launch ─────────────────────────────────────────────────

log "Launching agent..."
screen -dmS ralph-nigel "$TREE/.run-agent.sh"

echo ""
echo "  ┌──────────────────────────────────────────────┐"
echo "  │  Nigel agent running                          │"
echo "  │  Hub:       $HUB                              │"
echo "  │  Dashboard: $HUB/dashboard                    │"
echo "  │  Agent:     screen -r ralph-nigel             │"
echo "  └──────────────────────────────────────────────┘"
echo ""
echo "  Monitor:"
echo "    tail -f $TREE/agent.log"
echo "    curl -s $HUB/api/results/leaderboard | python3 -m json.tool"
echo ""
echo "  Stop:"
echo "    screen -S ralph-nigel -X quit"
echo ""

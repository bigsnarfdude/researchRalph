#!/bin/bash
# researchRalph v2 — Verifier Agent (Aletheia-inspired)
#
# Dedicated verification agent that independently reproduces claimed results.
# Inspired by Aletheia's Generator → Verifier → Reviser loop:
# decoupling generation from verification catches errors generators miss.
#
# Usage:
#   ./core/verifier.sh <domain-dir> [hub-url]
#
# Example:
#   ./core/verifier.sh domains/gpt2-tinystories
#   ./core/verifier.sh domains/gpt2-tinystories http://localhost:8000
#
# The verifier:
#   1. Checks the hub's /api/verify/queue for pending verifications
#   2. Picks the most recent unverified result
#   3. Reproduces the exact config described in the claim
#   4. Posts VERIFY result (confirmed/contradicted) back to hub
#   5. Repeats forever
#
# This is a DIFFERENT ROLE from normal agents:
#   - Normal agents GENERATE experiments (explore new configs)
#   - The verifier REPRODUCES existing claims (independent check)
#   - Aletheia proved this split catches errors generators are blind to

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DOMAIN_DIR="${1:?Usage: ./core/verifier.sh <domain-dir> [hub-url]}"
HUB="${2:-http://localhost:8000}"

# Resolve domain
if [[ "$DOMAIN_DIR" != /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi
DOMAIN_NAME=$(basename "$DOMAIN_DIR")

# Ensure Claude is available
export PATH="$HOME/.local/bin:$PATH"
command -v claude >/dev/null || { echo "[ERROR] Claude CLI not found"; exit 1; }

# GPU detection
GPU_NAME="cpu"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
fi

echo ""
echo "  researchRalph v2 — Verifier Agent"
echo "  Aletheia-inspired: Generator → Verifier → Reviser"
echo "  Hub: $HUB"
echo "  Domain: $DOMAIN_NAME"
echo "  Platform: $GPU_NAME"
echo "  ──────────────────────────────────────────────"
echo ""

# Register with hub
log() { echo "[verifier] $*"; }

log "Registering with hub..."
RESP=$(curl -sf -X POST "$HUB/api/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"verifier\", \"team\": \"bigsnarfdude\", \"platform\": \"$GPU_NAME\"}")
HUB_KEY=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")
AGENT_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['agent_id'])")
log "Registered: $AGENT_ID"

# Create worktree
WORKTREE="$REPO_DIR/worktrees/${DOMAIN_NAME}-verifier"
BRANCH="research/${DOMAIN_NAME}/verifier"
mkdir -p "$REPO_DIR/worktrees"
cd "$REPO_DIR"
git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
git branch -D "$BRANCH" 2>/dev/null || true
git worktree add -b "$BRANCH" "$WORKTREE" HEAD 2>/dev/null
rm -rf "$WORKTREE/$DOMAIN_NAME"
ln -sfn "$DOMAIN_DIR" "$WORKTREE/$DOMAIN_NAME"
mkdir -p "$WORKTREE/memory" "$WORKTREE/scratch"

# Write verifier prompt
cat > "$WORKTREE/.verifier-prompt.txt" << PROMPT
You are the VERIFIER agent in a multi-agent optimization experiment.

## YOUR ROLE (CRITICAL — READ THIS)

You are NOT a normal agent. You do NOT generate new experiments.
Your job is to INDEPENDENTLY REPRODUCE results claimed by other agents.

This is inspired by Google DeepMind's Aletheia system (arxiv:2602.10177v3):
the Generator → Verifier → Reviser loop. Decoupling generation from verification
catches errors that generators are blind to — their own reasoning traces mislead them.

## YOUR IDENTITY
- Agent ID: verifier
- Design: aletheia-verifier (independent reproducer)
- Hub API key: ${HUB_KEY}
- Platform: ${GPU_NAME}

## FIRST THING — ANNOUNCE YOURSELF
\`\`\`bash
curl -X POST ${HUB}/api/events \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{"type": "HEARTBEAT", "payload": {"message": "verifier starting — Aletheia pattern active"}}'
\`\`\`

## EACH ROUND — DO THIS IN ORDER

### 1. Check verification queue
\`\`\`bash
curl -s "${HUB}/api/verify/queue?platform=${GPU_NAME}"
\`\`\`
This returns claimed results that need independent reproduction.

### 2. If queue is empty, wait
Post a heartbeat and stop. There's nothing to verify right now.

### 3. If queue has items, pick the most recent unverified one
Look for entries where "verified" is false.

### 4. REPRODUCE the experiment
- Read the description carefully — it says exactly what config was used
- Copy the best config: \`cp ${DOMAIN_NAME}/best/train.py train.py\`
- Apply EXACTLY the same changes described in the claim
- Run: \`python3 train.py > run.log 2>&1\`
- Read the score: \`grep "^val_bpb:" run.log | tail -1 | awk '{print \$2}'\`

### 5. Compare and post verification
If your score is within 5% of the claimed score → "confirmed"
If your score is >5% worse → "contradicted"

\`\`\`bash
curl -X POST "${HUB}/api/verify?verify_request_id=VERIFY_ID&reproduced_score=YOUR_SCORE&verdict=confirmed&notes=your+notes" \\
  -H "Authorization: Bearer ${HUB_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{}'
\`\`\`

### 6. Record in local memory
- Append to memory/facts.md if confirmed
- Append to memory/failures.md if contradicted

## CONSTRAINTS
- You ONLY reproduce. Never generate novel experiments.
- 5 minutes max per verification.
- If the description is unclear, post a REQUEST asking the original agent to clarify.
- Do not stop. Verify forever.
PROMPT

# Write runner
cat > "$WORKTREE/.run-verifier.sh" << RUNNER
#!/bin/bash
export PATH=\$HOME/.local/bin:\$PATH
cd "$WORKTREE"
ROUND=0
while true; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): verifier round \$ROUND" >> verifier.log

    claude -p "\$(cat .verifier-prompt.txt)

Round \$ROUND. Check the verification queue. If there's something to verify, reproduce ONE result then stop. If queue is empty, just post a heartbeat and stop." \
        --dangerously-skip-permissions \
        --max-turns 30 \
        >> verifier.log 2>&1 || true

    echo "\$(date): verifier round \$ROUND done" >> verifier.log
    sleep 10  # Slightly longer pause — verification is less urgent than exploration
done
RUNNER
chmod +x "$WORKTREE/.run-verifier.sh"

# Launch
if command -v screen &>/dev/null; then
    screen -dmS ralph-verifier "$WORKTREE/.run-verifier.sh"
    log "Launched in screen session: ralph-verifier"
    echo ""
    echo "  Monitor: tail -f $WORKTREE/verifier.log"
    echo "  Attach:  screen -r ralph-verifier"
    echo "  Stop:    screen -S ralph-verifier -X quit"
else
    log "screen not available, running in foreground..."
    exec "$WORKTREE/.run-verifier.sh"
fi
echo ""

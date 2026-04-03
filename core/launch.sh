#!/bin/bash
# researchRalph v2 — Launch N agents with blackboard collaboration
#
# Usage:
#   ./core/launch.sh <domain-dir> [num-agents] [--gpu]
#
# Examples:
#   ./core/launch.sh domains/gpt2-tinystories 4
#   ./core/launch.sh domains/gpt2-tinystories 8 --gpu    # 1 GPU per agent
#   ./core/launch.sh domains/af-elicitation 8
#
# Prerequisites:
#   - Claude Code CLI installed and authenticated
#   - screen for session management
#   - git for worktree isolation

set -euo pipefail

# Model override: RRMA_MODEL=haiku ./core/launch.sh ...
MODEL_FLAG=""
if [ -n "${RRMA_MODEL:-}" ]; then
    MODEL_FLAG="--model $RRMA_MODEL"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/launch.sh <domain-dir> [num-agents] [--gpu]}"
NUM_AGENTS="${2:-4}"
GPU_MODE=false

# Resolve domain dir to absolute path
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

# Parse optional flags
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_MODE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
WORKTREE_DIR="$REPO_DIR/worktrees"
SESSION_PREFIX="ralph-${DOMAIN_NAME}"

echo "=== researchRalph v2 — Multi-Agent Launch ==="
echo "Domain:    $DOMAIN_DIR"
echo "Agents:    $NUM_AGENTS"
echo "GPU mode:  $GPU_MODE"
echo ""

# --- Validate domain ---
for required in program.md; do
    if [ ! -f "$DOMAIN_DIR/$required" ]; then
        echo "ERROR: Missing $DOMAIN_DIR/$required"
        echo "See domains/template/ for required files."
        exit 1
    fi
done

# --- Guard against re-launch ---
if screen -ls 2>/dev/null | grep -q "$SESSION_PREFIX"; then
    echo "ERROR: $SESSION_PREFIX screens already running. Stop them first:"
    echo "  ./core/stop.sh $DOMAIN_NAME"
    exit 1
fi

# --- GPU detection (if needed) ---
GPU_COUNT=0
if $GPU_MODE; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
        echo "GPUs detected: $GPU_COUNT"
    else
        echo "WARNING: --gpu specified but nvidia-smi not found. Agents will share GPU 0."
        GPU_COUNT=1
    fi
fi

# --- Initialize shared state ---
mkdir -p "$DOMAIN_DIR/queue" "$DOMAIN_DIR/active" "$DOMAIN_DIR/done" "$DOMAIN_DIR/best"

# Initialize results.tsv
if [ ! -s "$DOMAIN_DIR/results.tsv" ] || [ "$(wc -l < "$DOMAIN_DIR/results.tsv")" -le 1 ]; then
    printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$DOMAIN_DIR/results.tsv"
fi

# Initialize blackboard
if [ ! -s "$DOMAIN_DIR/blackboard.md" ]; then
cat > "$DOMAIN_DIR/blackboard.md" << 'EOF'
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
if [ ! -s "$DOMAIN_DIR/strategy.md" ]; then
cat > "$DOMAIN_DIR/strategy.md" << 'EOF'
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

# --- Create worktrees ---
mkdir -p "$WORKTREE_DIR"

DESIGNS=("vanilla" "memory" "blackboard" "blackboard" "blackboard" "blackboard" "blackboard" "blackboard")
# Agent 0 = vanilla (control), Agent 1 = memory, Agent 2+ = blackboard

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    BRANCH="research/${DOMAIN_NAME}/agent${AGENT}"
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT}"

    if [ -d "$TREE" ]; then
        git -C "$REPO_DIR" worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
    fi
    git -C "$REPO_DIR" branch -D "$BRANCH" 2>/dev/null || true

    echo "Creating worktree agent${AGENT}..."
    git -C "$REPO_DIR" worktree add -b "$BRANCH" "$TREE" HEAD

    # CRITICAL: symlink must REPLACE, not nest
    rm -rf "$TREE/$(basename "$DOMAIN_DIR")"
    ln -sfn "$DOMAIN_DIR" "$TREE/$(basename "$DOMAIN_DIR")"
    touch "$TREE/run.log"

    # Create agent memory structure
    mkdir -p "$TREE/memory" "$TREE/scratch"

    # Determine design
    DESIGN="blackboard"
    if [ "$AGENT" -eq 0 ]; then
        DESIGN="vanilla"
    elif [ "$AGENT" -eq 1 ]; then
        DESIGN="memory"
    fi

    # GPU assignment
    GPU_ENV=""
    if $GPU_MODE && [ "$GPU_COUNT" -gt 0 ]; then
        GPU=$((AGENT % GPU_COUNT))
        GPU_ENV="export CUDA_VISIBLE_DEVICES=$GPU"
    fi

    # Write agent prompt
    cat > "$TREE/.agent-prompt.txt" << PROMPT
You are agent $AGENT (design: $DESIGN) in a multi-agent optimization experiment.

Read $(basename "$DOMAIN_DIR")/program.md for the full protocol.

DESIGN: $DESIGN
$(if [ "$DESIGN" = "vanilla" ]; then
    echo "No memory. No collaboration. Just results.tsv and your judgment."
    echo "Record in results.tsv with agent${AGENT} and design=vanilla."
elif [ "$DESIGN" = "memory" ]; then
    echo "Full persistent memory. Maintain progress.md and next_ideas.md in your working dir."
    echo "Re-rank ideas after every experiment."
    echo "Record in results.tsv with agent${AGENT} and design=memory."
else
    echo "Structured memory (facts/failures/hunches) + shared blackboard + prediction tracking."
    echo ""
    echo "YOUR FILES (create if missing):"
    echo "- memory/facts.md: confirmed findings (append-only)"
    echo "- memory/failures.md: dead ends — NEVER retry"
    echo "- memory/hunches.md: suspicions worth testing"
    echo "- scratch/hypothesis.md: current theory + what you're testing"
    echo "- scratch/predictions.md: predicted vs actual score"
    echo ""
    echo "SHARED FILES (read AND write):"
    echo "- $(basename "$DOMAIN_DIR")/blackboard.md: post CLAIM, RESPONSE, REQUEST"
    echo "- $(basename "$DOMAIN_DIR")/results.tsv: append results"
    echo "- $(basename "$DOMAIN_DIR")/strategy.md: update when you become coordinator"
    echo ""
    echo "BLACKBOARD PROTOCOL:"
    echo "- CLAIM agent${AGENT}: <finding> (evidence: <experiment_id>, <metric>)"
    echo "- RESPONSE agent${AGENT} to agentM: <confirm/refute> — <reasoning>"
    echo "- REQUEST agent${AGENT} to agentM|any: <what to test>"
    echo ""
    echo "Record in results.tsv with agent${AGENT} and design=blackboard."
fi)

$([ -n "$GPU_ENV" ] && echo "GPU: $GPU_ENV")

CONSTRAINTS:
- Append to results.tsv with >> (never overwrite)
- Always start from: cp $(basename "$DOMAIN_DIR")/best/* .  (then apply changes)
- Do not stop. Do not ask questions. Run experiments forever.
PROMPT

    # Write runner script
    cat > "$TREE/.run-agent.sh" << RUNNER
#!/bin/bash
AGENT_ID=$AGENT
TREE_DIR="$TREE"
cd "\$TREE_DIR"
$GPU_ENV

MAX_ROUNDS=\${RRMA_MAX_ROUNDS:-3}
ROUND=0
while [ "\$ROUND" -lt "\$MAX_ROUNDS" ]; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): agent \$AGENT_ID starting round \$ROUND/\$MAX_ROUNDS" >> agent.log

    claude $MODEL_FLAG -p "\$(cat .agent-prompt.txt)

This is round \$ROUND. Check $(basename "$DOMAIN_DIR")/results.tsv for latest state. Continue the experiment loop." \
        --dangerously-skip-permissions \
        --max-turns 200 \
        >> agent.log 2>&1 || true

    echo "\$(date): agent \$AGENT_ID finished round \$ROUND/\$MAX_ROUNDS" >> agent.log
    sleep 5
done
echo "\$(date): agent \$AGENT_ID reached MAX_ROUNDS=\$MAX_ROUNDS, exiting." >> agent.log
RUNNER
    chmod +x "$TREE/.run-agent.sh"
done

echo ""
echo "=== Launching $NUM_AGENTS agents (staggered 30s apart) ==="
echo ""

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT}"
    SESSION="${SESSION_PREFIX}-agent${AGENT}"

    screen -S "$SESSION" -X quit 2>/dev/null || true
    screen -dmS "$SESSION" "$TREE/.run-agent.sh"

    DESIGN="blackboard"
    [ "$AGENT" -eq 0 ] && DESIGN="vanilla"
    [ "$AGENT" -eq 1 ] && DESIGN="memory"

    echo "  Agent $AGENT: design=$DESIGN screen=$SESSION"

    # Stagger launches
    if [ "$AGENT" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 30
    fi
done

echo ""
echo "=== $NUM_AGENTS agents launched ==="
echo ""
echo "Monitor:"
echo "  screen -ls                                              # list sessions"
echo "  screen -r ${SESSION_PREFIX}-agent0                      # attach (Ctrl+A D detach)"
echo "  cat $DOMAIN_DIR/results.tsv                             # all results"
echo "  cat $DOMAIN_DIR/blackboard.md                           # collaboration"
echo "  cat $DOMAIN_DIR/strategy.md                             # search strategy"
echo "  watch -n 30 'tail -20 $DOMAIN_DIR/results.tsv'          # live dashboard"
echo ""
echo "Stop:"
echo "  ./core/stop.sh $DOMAIN_NAME"
echo ""
echo "Collect:"
echo "  ./core/collect.sh $DOMAIN_NAME"

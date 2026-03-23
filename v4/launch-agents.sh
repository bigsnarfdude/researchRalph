#!/bin/bash
# launch-agents.sh — starts N worker agents + 1 meta-agent in screen sessions
#
# Usage: bash launch-agents.sh /path/to/domain [num_agents] [max_turns] [meta_interval_min]

DOMAIN_DIR="${1:-.}"
NUM_AGENTS="${2:-4}"
MAX_TURNS="${3:-200}"
META_INTERVAL="${4:-30}"

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find claude CLI
source "$SCRIPT_DIR/env.sh"
CLAUDE_BIN="$(command -v claude)"

echo "Domain: $DOMAIN_DIR"
echo "Workers: $NUM_AGENTS"
echo "Max turns per worker: $MAX_TURNS"
echo "Meta-agent interval: ${META_INTERVAL}m"
echo "Claude: $CLAUDE_BIN"
echo ""

# Check required files (flexible — not all domains have sae.py or engine.py)
for f in program.md blackboard.md run.sh; do
    if [ ! -f "$DOMAIN_DIR/$f" ]; then
        echo "Error: missing $DOMAIN_DIR/$f"
        exit 1
    fi
done

# Ensure results.tsv exists
touch "$DOMAIN_DIR/results.tsv"

echo "Files OK. Launching..."
echo ""

# Build PATH export for screen sessions
CLAUDE_DIR="$(dirname "$CLAUDE_BIN")"
EXTRA_PATH="$CLAUDE_DIR:$HOME/.local/bin"

# --- Launch worker agents ---
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    SESSION="rrma-worker$i"
    screen -S "$SESSION" -X quit 2>/dev/null

    screen -dmS "$SESSION" bash -c "
        export PATH=\"$EXTRA_PATH:\$PATH\"
        cd $DOMAIN_DIR
        export AGENT_ID=agent$i
        claude -p 'You are agent$i. Read program.md, blackboard.md, results.tsv, and best/. If meta-blackboard.md exists, read it — it contains compressed observations from previous cycles. If calibration.md exists, read it — it contains known results and techniques from the literature. Then start experimenting. Write all findings to blackboard.md. Periodically re-read meta-blackboard.md — it updates during the run. Never stop. IMPORTANT: Only read files in the current directory. Do not read files from other domains or directories in this repository.' \
            --dangerously-skip-permissions \
            --max-turns $MAX_TURNS
    "
    echo "Started $SESSION (screen -r $SESSION)"

    # Stagger launches to avoid resource contention
    if [ "$i" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 15
    fi
done

# --- Launch meta-agent ---
SESSION="rrma-meta"
screen -S "$SESSION" -X quit 2>/dev/null

screen -dmS "$SESSION" bash -c "
    export PATH=\"$EXTRA_PATH:\$PATH\"
    bash $SCRIPT_DIR/meta-loop.sh $DOMAIN_DIR $META_INTERVAL
"
echo "Started $SESSION (screen -r $SESSION)"

echo ""
echo "All running. Monitor with:"
echo "  screen -ls                          # list sessions"
echo "  screen -r rrma-worker0              # attach to worker 0"
echo "  screen -r rrma-meta                 # attach to meta-agent"
echo "  tail -f $DOMAIN_DIR/results.tsv     # watch scores"
echo "  cat $DOMAIN_DIR/meta-blackboard.md  # read meta reflections"
echo ""
echo "To stop everything:"
echo "  bash $SCRIPT_DIR/stop-agents.sh"

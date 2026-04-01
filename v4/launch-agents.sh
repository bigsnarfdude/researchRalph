#!/bin/bash
# launch-agents.sh — starts N worker agents + 1 meta-agent in screen sessions (v4.6)
#
# v4.6 changes:
#   - Agents read program_static.md (once) + program.md (dynamic) instead of monolithic program.md
#   - Agents read stoplight.md + recent_experiments.md instead of full blackboard
#   - refresh_context.py generates stoplight + recent_experiments before launch
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

# Ensure results.tsv and logs dir exist
touch "$DOMAIN_DIR/results.tsv"
mkdir -p "$DOMAIN_DIR/logs"

# Rotate any existing agent logs — clean naming: agent0_s1.jsonl, agent0_s2.jsonl
for existing in "$DOMAIN_DIR/logs"/agent*.jsonl; do
    [ -f "$existing" ] || continue
    base="$(basename "$existing" .jsonl)"
    # Extract agent prefix (agent0, agent1, etc.)
    agent_prefix=$(echo "$base" | grep -oE '^agent[0-9]+')
    [ -z "$agent_prefix" ] && agent_prefix="$base"
    # Find next session number for this agent
    next_s=$(ls "$DOMAIN_DIR/logs/${agent_prefix}_s"*.jsonl 2>/dev/null | grep -oE '_s[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
    next_s=$(( ${next_s:-0} + 1 ))
    mv "$existing" "$DOMAIN_DIR/logs/${agent_prefix}_s${next_s}.jsonl"
    echo "Rotated: $(basename "$existing") → ${agent_prefix}_s${next_s}.jsonl"
done

echo "Files OK."

# --- v4.6: Generate initial context files ---
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/tools/refresh_context.py" ]; then
    python3 "$REPO_ROOT/tools/refresh_context.py" "$DOMAIN_DIR" 2>&1
fi

# --- v4.7: Create agent-local workspaces ---
# Each agent gets workspace/agentN/ with its own copy of train.py
# Eliminates race condition where agents overwrite each other's train.py
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    WS="$DOMAIN_DIR/workspace/agent$i"
    mkdir -p "$WS"
    # Seed from best/train.py if available, else domain root train.py
    if [ -f "$DOMAIN_DIR/best/train.py" ]; then
        cp "$DOMAIN_DIR/best/train.py" "$WS/train.py"
    elif [ -f "$DOMAIN_DIR/train.py" ]; then
        cp "$DOMAIN_DIR/train.py" "$WS/train.py"
    fi
    echo "Workspace: workspace/agent$i/train.py ready"
done

echo "Launching..."
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
        export CLAUDE_AGENT_ID=agent$i
        claude --output-format stream-json --verbose \
            --dangerously-skip-permissions \
            --max-turns $MAX_TURNS \
            -p 'You are agent$i. Read these files in order:

1. program_static.md — immutable rules, harness protocol, scoring, lifecycle (read ONCE, do not re-read)
2. program.md — dynamic guidance, current regime, closed brackets, constraints (re-read when stuck)
3. stoplight.md — compressed run state: health, what works, dead ends, recent activity
4. recent_experiments.md — last 5 experiments with structured outcomes + full score trajectory
5. best/train.py — current best config (READ ONLY — do not edit best/ directly)
6. If meta-blackboard.md exists, read it — compressed observations from previous cycles.
7. If calibration.md exists, read it — known results from the literature.

If program_static.md does not exist, read program.md for everything (backwards compatibility).
If stoplight.md does not exist, read blackboard.md instead.

## YOUR WORKSPACE (v4.7 — no more race conditions)
Your private workspace is: workspace/agent$i/
- Copy best/train.py → workspace/agent$i/train.py at the start of each experiment cycle
- Edit ONLY workspace/agent$i/train.py — never edit train.py in the domain root or best/
- run.sh automatically picks up workspace/agent$i/train.py when you run it
- Other agents cannot see or modify your workspace

Workflow per experiment:
  cp best/train.py workspace/agent$i/train.py
  # make your ONE change to workspace/agent$i/train.py
  bash run.sh <name> "description" <design_type>

Then start experimenting. Write all findings to blackboard.md. Periodically re-read stoplight.md and recent_experiments.md — they update during the run. After every experiment append to: MISTAKES.md (tactics that failed and why), DESIRES.md (tools or context you wish you had), LEARNINGS.md (discoveries about the environment). Never stop. IMPORTANT: Only read files in the current directory. Do not read files from other domains or directories in this repository.' \
            > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1
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

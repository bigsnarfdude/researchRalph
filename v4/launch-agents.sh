#!/bin/bash
# launch-agents.sh — starts N worker agents + 1 meta-agent in screen sessions (v4.6)
#
# v4.6 changes:
#   - Agents read program_static.md (once) + program.md (dynamic) instead of monolithic program.md
#   - Agents read stoplight.md + recent_experiments.md instead of full blackboard
#   - refresh_context.py generates stoplight + recent_experiments before launch
#
# Usage: bash launch-agents.sh /path/to/domain [num_agents] [max_turns] [meta_interval_min] [model]

DOMAIN_DIR="${1:-.}"
NUM_AGENTS="${2:-4}"
MAX_TURNS="${3:-200}"
META_INTERVAL="${4:-30}"
MODEL="${5:-}"  # e.g. claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
# Read editable file from config.yaml (lean_proof uses *.lean, ML uses train.py)
EDITABLE_FILE="$(grep '^editable:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE_FILE="${EDITABLE_FILE:-train.py}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find claude CLI
source "$SCRIPT_DIR/env.sh"
CLAUDE_BIN="$(command -v claude)"

# Screen session prefix — set RRMA_PREFIX to run concurrent fleets on one host
PREFIX="${RRMA_PREFIX:-rrma}"

# v4.9: Resolve the worker workflow section by domain type.
# Priority: domain-local worker_prompt.md > v4/prompts/<domain_type>.md > ml_default.md
# This is what keeps a Lean workflow out of ML agents' prompts (and vice versa).
DOMAIN_TYPE="$(grep '^domain_type:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
if [ -f "$DOMAIN_DIR/worker_prompt.md" ]; then
    WORKFLOW_TEMPLATE="$DOMAIN_DIR/worker_prompt.md"
elif [ -n "$DOMAIN_TYPE" ] && [ -f "$SCRIPT_DIR/prompts/$DOMAIN_TYPE.md" ]; then
    WORKFLOW_TEMPLATE="$SCRIPT_DIR/prompts/$DOMAIN_TYPE.md"
elif [ -f "$SCRIPT_DIR/prompts/ml_default.md" ]; then
    WORKFLOW_TEMPLATE="$SCRIPT_DIR/prompts/ml_default.md"
    echo "Warning: no workflow template for domain_type='${DOMAIN_TYPE:-unset}' — using ml_default.md"
else
    echo "Error: no worker workflow template found (worker_prompt.md or $SCRIPT_DIR/prompts/)"
    exit 1
fi

echo "Domain: $DOMAIN_DIR"
echo "Workers: $NUM_AGENTS"
echo "Max turns per worker: $MAX_TURNS"
echo "Meta-agent interval: ${META_INTERVAL}m"
echo "Claude: $CLAUDE_BIN"
echo "Model: ${MODEL:-default}"
echo "Session prefix: $PREFIX"
echo "Workflow template: $WORKFLOW_TEMPLATE"
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

# --- v4.8: Seed domain memory if missing ---
if [ ! -d "$DOMAIN_DIR/memory" ]; then
    python3 "$REPO_ROOT/tools/memory_system.py" seed "$DOMAIN_DIR" 2>&1
fi

# --- v4.7: Create agent-local workspaces ---
# Each agent gets workspace/agentN/ with its own copy of train.py
# Eliminates race condition where agents overwrite each other's train.py
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    WS="$DOMAIN_DIR/workspace/agent$i"
    mkdir -p "$WS"
    # Seed workspace with editable file (train.py for ML, *.lean for Lean domains)
    if [ -f "$DOMAIN_DIR/best/$EDITABLE_FILE" ]; then
        cp "$DOMAIN_DIR/best/$EDITABLE_FILE" "$WS/$EDITABLE_FILE"
    elif [ -f "$DOMAIN_DIR/$EDITABLE_FILE" ]; then
        cp "$DOMAIN_DIR/$EDITABLE_FILE" "$WS/$EDITABLE_FILE"
    fi
    echo "Workspace: workspace/agent$i/$EDITABLE_FILE ready"
done

echo "Launching..."
echo ""

# Build PATH export for screen sessions
CLAUDE_DIR="$(dirname "$CLAUDE_BIN")"
EXTRA_PATH="$CLAUDE_DIR:$HOME/.local/bin"

# --- Launch worker agents ---
mkdir -p "$DOMAIN_DIR/.agent_prompts"

for i in $(seq 0 $((NUM_AGENTS - 1))); do
    SESSION="${PREFIX}-worker$i"
    screen -S "$SESSION" -X quit 2>/dev/null

    # v4.8: Pre-generate verified memory context for this agent
    MEMORY_CONTEXT=""
    if [ -d "$DOMAIN_DIR/memory" ]; then
        MEMORY_CONTEXT=$(python3 "$REPO_ROOT/tools/memory_system.py" --json recall \
            "$DOMAIN_DIR/memory/" "agent$i startup: current best, closed brackets, key findings" \
            --domain-dir "$DOMAIN_DIR" --top 5 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for fname, content in data.get('content', {}).items():
        # Strip frontmatter, keep only body + any verification tags
        lines = content.split('\n')
        in_fm = False
        body = []
        for line in lines:
            if line.strip() == '---':
                in_fm = not in_fm
                continue
            if not in_fm:
                body.append(line)
        print('\n'.join(body).strip())
        print()
except: pass
" 2>/dev/null)
    fi

    # v4.9: Assemble the worker prompt from universal scaffolding + the domain-type
    # workflow template. Written to a file so it's auditable and the screen command
    # avoids quoting hell.
    PROMPT_FILE="$DOMAIN_DIR/.agent_prompts/agent$i.md"
    {
        cat <<UNIVERSAL_EOF
You are agent$i. Read these files in order:

1. program_static.md — immutable rules, harness protocol, scoring, lifecycle (read ONCE, do not re-read)
2. program.md — dynamic guidance, current regime, closed brackets, constraints (re-read when stuck)
3. stoplight.md — compressed run state: health, what works, dead ends, recent activity
4. recent_experiments.md — last 5 experiments with structured outcomes + full score trajectory
5. If best/$EDITABLE_FILE exists, read it — current best proof/config (READ ONLY)
6. If meta-blackboard.md exists, read it — compressed observations from previous cycles.
7. If calibration.md exists, read it — known results from the literature.
8. Re-read blackboard.md every 5 oracle calls — the overseer drops hints mid-run. If workspace/agent$i/BLACKBOARD_UPDATED exists, re-read blackboard.md immediately, then delete that flag file.

## Verified Memory (auto-loaded, checked against live sources)
${MEMORY_CONTEXT:-No domain memory available.}

If program_static.md does not exist, read program.md for everything (backwards compatibility).
If stoplight.md does not exist, read blackboard.md instead.


## REWARD HACKING PREVENTION — READ THIS
results.tsv is write-protected (chmod 444). Direct writes will fail with permission denied.
The ONLY way to register an experiment is: bash run.sh
Never write to results.tsv yourself. Never claim a score in blackboard.md without citing exact run.sh output.
If run.sh returns SCORE=0.0, that is the score. Do not report otherwise.
Unverified claims in blackboard.md will be flagged by the oracle audit.
## YOUR WORKSPACE (v4.7 — no more race conditions)
Your private workspace is: workspace/agent$i/
- Your editable file is: workspace/agent$i/$EDITABLE_FILE
- Edit ONLY your workspace copy — never edit the domain root or best/ directly
- run.sh automatically picks up your workspace file (CLAUDE_AGENT_ID is set for you)
- Other agents cannot see or modify your workspace

UNIVERSAL_EOF
        sed -e "s|{{AGENT_ID}}|agent$i|g" -e "s|{{EDITABLE_FILE}}|$EDITABLE_FILE|g" "$WORKFLOW_TEMPLATE"
        cat <<'FOOTER_EOF'

Then start experimenting. Write all findings to blackboard.md. Periodically re-read stoplight.md and recent_experiments.md — they update during the run. After every experiment append to: MISTAKES.md (tactics that failed and why), DESIRES.md (tools or context you wish you had), LEARNINGS.md (discoveries about the environment). Never stop. IMPORTANT: Only read files in the current directory. Do not read files from other domains or directories in this repository.
FOOTER_EOF
    } > "$PROMPT_FILE"

    screen -dmS "$SESSION" bash -c "
        export PATH=\"$EXTRA_PATH:\$PATH\"
        cd $DOMAIN_DIR
        export AGENT_ID=agent$i
        export CLAUDE_AGENT_ID=agent$i
        claude --output-format stream-json --verbose \
            --dangerously-skip-permissions \
            ${MODEL:+--model $MODEL} \
            --max-turns $MAX_TURNS \
            -p \"\$(cat $PROMPT_FILE)\" \
            > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1
    "
    echo "Started $SESSION (screen -r $SESSION)"

    # Stagger launches to avoid resource contention
    if [ "$i" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 15
    fi
done

# --- Launch meta-agent ---
SESSION="${PREFIX}-meta"
screen -S "$SESSION" -X quit 2>/dev/null

screen -dmS "$SESSION" bash -c "
    export PATH=\"$EXTRA_PATH:\$PATH\"
    bash $SCRIPT_DIR/meta-loop.sh $DOMAIN_DIR $META_INTERVAL
"
echo "Started $SESSION (screen -r $SESSION)"

echo ""
echo "All running. Monitor with:"
echo "  screen -ls                          # list sessions"
echo "  screen -r ${PREFIX}-worker0         # attach to worker 0"
echo "  screen -r ${PREFIX}-meta            # attach to meta-agent"
echo "  tail -f $DOMAIN_DIR/results.tsv     # watch scores"
echo "  cat $DOMAIN_DIR/meta-blackboard.md  # read meta reflections"
echo ""
echo "To stop everything:"
echo "  bash $SCRIPT_DIR/stop-agents.sh"

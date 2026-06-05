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
PREFIX="${6:-worker}"

DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
# Read editable file from config.yaml (lean_proof uses *.lean, ML uses train.py)
EDITABLE_FILE="$(grep '^editable:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE_FILE="${EDITABLE_FILE:-train.py}"
DOMAIN_TYPE="$(grep '^domain_type:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
DOMAIN_TYPE="${DOMAIN_TYPE:-ml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find claude CLI
source "$SCRIPT_DIR/env.sh"
CLAUDE_BIN="$(command -v claude)"

echo "Domain: $DOMAIN_DIR"
echo "Workers: $NUM_AGENTS"
echo "Max turns per worker: $MAX_TURNS"
echo "Meta-agent interval: ${META_INTERVAL}m"
echo "Claude: $CLAUDE_BIN"
echo "Model: ${MODEL:-default}"
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
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    SESSION="rrma-${PREFIX}-w$i"
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

    # Build the agent prompt — Lean domains use a slim, edit-first prompt
    if [ "$DOMAIN_TYPE" = "lean_proof" ]; then
        AGENT_PROMPT="You are agent$i for a Lean 4 proof domain.

REQUIRED first reads (ONCE, in this order):
1. program.md — task description and construction
2. mathlib_hints.md — exact Mathlib lemma names and tactic patterns

After those two reads, your IMMEDIATE next action must be Edit on workspace/agent$i/$EDITABLE_FILE.
Do NOT read any other file. No blackboard, no stoplight, no calibration, no LEARNINGS, no other agents.

Your workspace: workspace/agent$i/$EDITABLE_FILE (already seeded with theorem statement + sorry)

Loop:
  1. Edit workspace/agent$i/$EDITABLE_FILE — write or extend the proof
  2. Run: bash run.sh   (oracle compiles your file, prints SORRY_COUNT + errors + SCORE)
  3. Read compiler errors from stdout, fix in next Edit
  4. Goto 1 until SCORE=1.0

HARD RULES:
- After the initial two reads, never go more than 1 Edit without bash run.sh.
- results.tsv is read-only — never try to write it. Oracle logs scores automatically.
- Stay in your workspace. Do not read /home/vincent paths outside this domain dir.
- Do not call WebSearch, Glob other workspaces, or read random files.
- Iterate fast. Lean errors are precise — use them, do not over-think.

MANDATORY EFFORT BUDGET: This is a hard COLD problem with no construction given. You MUST genuinely attempt at least 6 DIFFERENT candidate constructions for the set A before you may conclude it is unsolvable. For EACH candidate: write the construction definitions plus as much of the proof as you can, then run bash run.sh to test it; log each distinct candidate in MISTAKES.md. Do NOT stop or write a give-up summary before testing 6 distinct candidates. NEVER fabricate, never leave the main theorem as sorry to claim success, never weaken the statement.

START NOW: Read program.md, then mathlib_hints.md, then Edit your workspace file."
    else
        AGENT_PROMPT="You are agent$i. Read these files in order:

1. program_static.md — immutable rules, harness protocol, scoring, lifecycle (read ONCE, do not re-read)
2. program.md — dynamic guidance, current regime, closed brackets, constraints (re-read when stuck)
3. stoplight.md — compressed run state: health, what works, dead ends, recent activity
4. recent_experiments.md — last 5 experiments with structured outcomes + full score trajectory
5. If best/$EDITABLE_FILE exists, read it — current best proof/config (READ ONLY)
6. If meta-blackboard.md exists, read it — compressed observations from previous cycles.
7. If calibration.md exists, read it — known results from the literature.
8. Re-read blackboard.md every 5 oracle calls — the overseer drops hints mid-run. If BLACKBOARD_UPDATED exists in the domain dir, re-read blackboard.md immediately before your next edit.

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

## WORKFLOW
Goal: improve workspace/agent$i/$EDITABLE_FILE; call bash run.sh after every edit.

Then start experimenting. Write all findings to blackboard.md. After every experiment append to: MISTAKES.md, DESIRES.md, LEARNINGS.md. Never stop. IMPORTANT: Only read files in the current directory."
    fi

    screen -dmS "$SESSION" bash -c "
        export PATH=\"$EXTRA_PATH:\$PATH\"
        cd $DOMAIN_DIR
        export AGENT_ID=agent$i
        export CLAUDE_AGENT_ID=agent$i
        if [ "$DOMAIN_TYPE" = "lean_proof" ]; then
            claude --dangerously-skip-permissions \
                ${MODEL:+--model $MODEL} \
                -p \"$AGENT_PROMPT\" \
                > $DOMAIN_DIR/logs/agent${i}.log 2>&1
        else
            claude --output-format stream-json --verbose \
                --dangerously-skip-permissions \
                ${MODEL:+--model $MODEL} \
                -p \"$AGENT_PROMPT\" \
                > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1
        fi
    "
    echo "Started $SESSION (screen -r $SESSION)"

    # Stagger launches to avoid resource contention
    if [ "$i" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 15
    fi
done

# --- Launch meta-agent ---
SESSION="rrma-${PREFIX}-meta"
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

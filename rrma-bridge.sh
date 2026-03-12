#!/bin/bash
# battleBOT ↔ RRMA Bridge
#
# Links a battleBOT game as an RRMA domain, then launches the swarm.
#
# Usage:
#   ./rrma-bridge.sh <game-name> [num-agents]
#
# Examples:
#   ./rrma-bridge.sh economy 4      # 4 agents optimize economy
#   ./rrma-bridge.sh arena 2        # 2 agents optimize arena
#   ./rrma-bridge.sh all 4          # run all 8 games sequentially
#
# Prerequisites:
#   - researchRalph-v2 at ~/Downloads/researchRalph-v2/
#   - Claude Code CLI installed and authenticated
#   - screen for session management

set -euo pipefail

BATTLEBOTDIR="$(cd "$(dirname "$0")" && pwd)"
RRMA_DIR="${RRMA_DIR:-$HOME/Downloads/researchRalph-v2}"
GAME="${1:?Usage: ./rrma-bridge.sh <game-name|all> [num-agents]}"
NUM_AGENTS="${2:-4}"

# --- Validate RRMA exists ---
if [ ! -d "$RRMA_DIR/core" ]; then
    echo "ERROR: researchRalph-v2 not found at $RRMA_DIR"
    echo "Set RRMA_DIR env var or ensure ~/Downloads/researchRalph-v2/ exists"
    exit 1
fi

# --- Handle "all" mode ---
if [ "$GAME" = "all" ]; then
    for game_dir in "$BATTLEBOTDIR"/games/*/; do
        game_name="$(basename "$game_dir")"
        echo ""
        echo "════════════════════════════════════════"
        echo "  Running swarm on: $game_name"
        echo "════════════════════════════════════════"
        "$0" "$game_name" "$NUM_AGENTS"
    done
    exit 0
fi

# --- Validate game exists ---
GAME_DIR="$BATTLEBOTDIR/games/$GAME"
if [ ! -d "$GAME_DIR" ]; then
    echo "ERROR: Game '$GAME' not found in $BATTLEBOTDIR/games/"
    echo "Available games:"
    ls -1 "$BATTLEBOTDIR/games/" 2>/dev/null || echo "  (none)"
    exit 1
fi

# Validate required files
for f in program.md config.yaml run.sh engine.py; do
    if [ ! -f "$GAME_DIR/$f" ]; then
        echo "ERROR: Missing $GAME_DIR/$f"
        exit 1
    fi
done

echo "=== battleBOT ↔ RRMA Bridge ==="
echo "Game:      $GAME"
echo "Game dir:  $GAME_DIR"
echo "Agents:    $NUM_AGENTS"
echo "RRMA:      $RRMA_DIR"
echo ""

# --- Prepare game as RRMA domain ---

# Create shared state directories
mkdir -p "$GAME_DIR/best"

# Copy default config as initial "best"
if [ ! -f "$GAME_DIR/best/config.yaml" ]; then
    cp "$GAME_DIR/config.yaml" "$GAME_DIR/best/config.yaml"
    echo "Initialized best/config.yaml from defaults"
fi

# Initialize results.tsv (only run baseline if needed)
if [ ! -s "$GAME_DIR/results.tsv" ] || [ "$(wc -l < "$GAME_DIR/results.tsv")" -le 1 ]; then
    echo "Running baseline..."
    BASELINE=$(bash "$GAME_DIR/run.sh" "$GAME_DIR/best/config.yaml" 2>/dev/null | tail -1)
    echo "Baseline score: $BASELINE"
    printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$GAME_DIR/results.tsv"
    printf 'baseline\t%s\t0\tkeep\tdefault config\tsetup\tbaseline\n' "$BASELINE" >> "$GAME_DIR/results.tsv"
    echo "Initialized results.tsv with baseline"
else
    BASELINE=$(awk 'NR==2{print $2}' "$GAME_DIR/results.tsv")
    echo "Baseline score: $BASELINE (from existing results.tsv)"
fi

# Initialize blackboard — just a shared lab notebook, no protocol
if [ ! -s "$GAME_DIR/blackboard.md" ]; then
cat > "$GAME_DIR/blackboard.md" << EOF
# Blackboard — $GAME

Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.

Baseline: $BASELINE (best/config.yaml)
EOF
    echo "Initialized blackboard.md"
fi

echo ""
echo "Game '$GAME' is now an RRMA-compatible domain."
echo ""

# --- Symlink into RRMA domains/ ---
RRMA_DOMAIN="$RRMA_DIR/domains/battlebotgym-$GAME"
if [ -L "$RRMA_DOMAIN" ] || [ -d "$RRMA_DOMAIN" ]; then
    rm -rf "$RRMA_DOMAIN"
fi
ln -sfn "$GAME_DIR" "$RRMA_DOMAIN"
echo "Symlinked: $RRMA_DOMAIN → $GAME_DIR"

# --- Guard against re-launch ---
SESSION_PREFIX="ralph-battlebotgym-${GAME}"
if screen -ls 2>/dev/null | grep -q "$SESSION_PREFIX"; then
    echo "ERROR: $SESSION_PREFIX screens already running. Stop them first:"
    echo "  $RRMA_DIR/core/stop.sh battlebotgym-$GAME"
    exit 1
fi

# --- Create worktrees ---
WORKTREE_DIR="$RRMA_DIR/worktrees"
mkdir -p "$WORKTREE_DIR"

DOMAIN_NAME="battlebotgym-$GAME"

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    BRANCH="research/${DOMAIN_NAME}/agent${AGENT}"
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT}"

    if [ -d "$TREE" ]; then
        git -C "$RRMA_DIR" worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
    fi
    git -C "$RRMA_DIR" branch -D "$BRANCH" 2>/dev/null || true

    echo "Creating worktree agent${AGENT}..."
    git -C "$RRMA_DIR" worktree add -b "$BRANCH" "$TREE" HEAD

    # CRITICAL: symlink must REPLACE, not nest
    rm -rf "$TREE/$DOMAIN_NAME"
    ln -sfn "$GAME_DIR" "$TREE/$DOMAIN_NAME"
    touch "$TREE/run.log"

    # Agent prompt — minimal. program.md has the domain details.
    cat > "$TREE/.agent-prompt.txt" << PROMPT
You are agent ${AGENT} working on the "$GAME" game. There are $NUM_AGENTS agents total.

Follow the instructions in ${DOMAIN_NAME}/program.md.

Before each experiment:
  1. Read ${DOMAIN_NAME}/blackboard.md and ${DOMAIN_NAME}/results.tsv — don't duplicate what others tried.
  2. Think about what to try and why. Read the code. Search for papers if relevant.

After each experiment:
  1. Append your result to ${DOMAIN_NAME}/results.tsv (>> only, never overwrite).
  2. Write what you tried, what happened, and WHY in ${DOMAIN_NAME}/blackboard.md.
  3. If new best, update ${DOMAIN_NAME}/best/.

Start from: cp ${DOMAIN_NAME}/best/config.yaml config.yaml
You may also edit source files (sae.py, engine.py, etc.) — the biggest wins come from code changes, not config tuning.
PROMPT

    # Runner script
    cat > "$TREE/.run-agent.sh" << RUNNER
#!/bin/bash
export PATH="\$HOME/.local/bin:\$PATH"
AGENT_ID=$AGENT
TREE_DIR="$TREE"
cd "\$TREE_DIR"

ROUND=0
while true; do
    ROUND=\$((ROUND + 1))
    echo "\$(date): agent \$AGENT_ID starting round \$ROUND" >> agent.log

    claude -p "\$(cat .agent-prompt.txt)

This is round \$ROUND. Check ${DOMAIN_NAME}/blackboard.md for what others have tried. Continue." \
        --dangerously-skip-permissions \
        --max-turns 200 \
        >> agent.log 2>&1 || true

    echo "\$(date): agent \$AGENT_ID exited round \$ROUND, restarting in 5s..." >> agent.log
    sleep 5
done
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

    echo "  Agent $AGENT: screen=$SESSION"

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
echo "  cat $GAME_DIR/results.tsv                               # all results"
echo "  cat $GAME_DIR/blackboard.md                             # shared lab notes"
echo "  watch -n 30 'tail -20 $GAME_DIR/results.tsv'            # live dashboard"
echo ""
echo "Stop:"
echo "  $RRMA_DIR/core/stop.sh battlebotgym-$GAME"
echo ""

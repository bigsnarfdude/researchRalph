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
mkdir -p "$GAME_DIR/best" "$GAME_DIR/queue" "$GAME_DIR/active" "$GAME_DIR/done"

# Copy default config as initial "best"
if [ ! -f "$GAME_DIR/best/config.yaml" ]; then
    cp "$GAME_DIR/config.yaml" "$GAME_DIR/best/config.yaml"
    echo "Initialized best/config.yaml from defaults"
fi

# Get baseline score
echo "Running baseline..."
BASELINE=$(bash "$GAME_DIR/run.sh" "$GAME_DIR/best/config.yaml" 2>/dev/null | tail -1)
echo "Baseline score: $BASELINE"

# Initialize results.tsv
if [ ! -s "$GAME_DIR/results.tsv" ] || [ "$(wc -l < "$GAME_DIR/results.tsv")" -le 1 ]; then
    printf 'commit\tscore\tmemory_gb\tstatus\tdescription\tagent\tdesign\n' > "$GAME_DIR/results.tsv"
    printf 'baseline\t%s\t0\tkeep\tdefault config\tsetup\tbaseline\n' "$BASELINE" >> "$GAME_DIR/results.tsv"
    echo "Initialized results.tsv with baseline"
fi

# Initialize blackboard
if [ ! -s "$GAME_DIR/blackboard.md" ]; then
cat > "$GAME_DIR/blackboard.md" << EOF
# Shared Blackboard — $GAME

Format: append claims, responses, and requests below.

## Claims
<!-- CLAIM agentN: finding (evidence: experiment_id, metric) -->

## Responses
<!-- RESPONSE agentN to agentM: confirmed/refuted (evidence) -->

## Requests
<!-- REQUEST agentN to agentM|any: what to test (priority: high|medium|low) -->
EOF
    echo "Initialized blackboard.md"
fi

# Initialize strategy
if [ ! -s "$GAME_DIR/strategy.md" ]; then
cat > "$GAME_DIR/strategy.md" << EOF
# Search Strategy — $GAME

## Current Best
- Score: $BASELINE
- Config: best/config.yaml

## Phase: exploration

## What works (high confidence)
(none yet — start exploring)

## What fails (avoid)
(none yet)

## Untested
(everything — read program.md for parameter space)
EOF
    echo "Initialized strategy.md"
fi

echo ""
echo "Game '$GAME' is now an RRMA-compatible domain."
echo ""

# --- Anti-herding role assignments ---
# Each agent gets a distinct search role to prevent convergence.
# Roles are free (strategy weights only) — they don't change the launch mechanics.

ROLES=(
    "SCOUT: You are the explorer. Try WILD, unconventional configs that nobody else would try. High variance is good. Flip assumptions — if the best config is defensive, try max offense. If everyone is tuning stats, focus on strategy weights. Post REQUEST for promising finds. Do NOT refine — move on after 2 experiments per idea."
    "EXPLOIT: You are the refiner. Start from the current best config and make small, systematic ±5-10% changes to ONE parameter at a time. Measure precisely. Your job is to hill-climb the local optimum. Never try radical changes."
    "DIVERSITY: You are the anti-herding agent. Read the blackboard and results.tsv, then deliberately try the OPPOSITE of what other agents are doing. If they're all high-aggression, try low. If they cluster on one archetype, try the least-tested one. Track which parameter regions are unexplored."
    "ANALYST: You are the pattern finder. Run experiments, but spend extra time analyzing results.tsv for interactions and nonlinear effects. Test combo hypotheses (e.g., 'high aggression + low retreat works but only with range >= 3'). Post detailed CLAIMs with evidence."
    "BERSERKER: You specialize in aggressive, high-attack builds. Explore the berserker/glass-cannon/assassin space. Push attack_power, special_power, and aggression to extremes. Find the best offensive build."
    "TURTLE: You specialize in defensive, survival builds. Explore turtle/brawler space. Push defense, heal_amount, max_hp. Find the best tank build that outlasts the champion."
    "SNIPER: You specialize in ranged kiting builds. Push attack_range, kite_distance, low aggression. Find the optimal hit-and-run strategy."
    "HYBRID: You test combinations that cross archetype boundaries. Try sniper+healer, brawler+burst, turtle+high-special. Look for synergies that pure archetypes miss."
)

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

# --- Create worktrees with role-specific prompts ---
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

    # Create agent memory structure
    mkdir -p "$TREE/memory" "$TREE/scratch"

    # Assign role (cycles if more agents than roles)
    ROLE_IDX=$((AGENT % ${#ROLES[@]}))
    ROLE="${ROLES[$ROLE_IDX]}"

    # Write role-specific agent prompt
    cat > "$TREE/.agent-prompt.txt" << PROMPT
You are agent $AGENT in a multi-agent optimization experiment on the "$GAME" game.

Read ${DOMAIN_NAME}/program.md for the full protocol.

DESIGN: blackboard (structured collaboration)

YOUR ROLE:
$ROLE

YOUR FILES (create if missing):
- memory/facts.md: confirmed findings (append-only)
- memory/failures.md: dead ends — NEVER retry
- memory/hunches.md: suspicions worth testing
- scratch/hypothesis.md: current theory + what you're testing
- scratch/predictions.md: predicted vs actual score

SHARED FILES (read AND write):
- ${DOMAIN_NAME}/blackboard.md: post CLAIM, RESPONSE, REQUEST
- ${DOMAIN_NAME}/results.tsv: append results
- ${DOMAIN_NAME}/strategy.md: update when you become coordinator

BLACKBOARD PROTOCOL:
- CLAIM agent${AGENT}: <finding> (evidence: <experiment_id>, <metric>)
- RESPONSE agent${AGENT} to agentM: <confirm/refute> — <reasoning>
- REQUEST agent${AGENT} to agentM|any: <what to test>

ANTI-HERDING RULES:
1. Before each experiment, read results.tsv. Do NOT test configs similar to what others already tested.
2. If your last 3 results are within 2% of each other, you are stuck. Try a completely different archetype.
3. Your role defines your search region. Stay in your lane unless you find strong evidence to cross over.
4. When posting CLAIMs, include which parameter region you explored so others avoid it.

Record in results.tsv with agent${AGENT} and design=blackboard.

CONSTRAINTS:
- Append to results.tsv with >> (never overwrite)
- Always start from: cp ${DOMAIN_NAME}/best/config.yaml config.yaml (then apply changes)
- Do not stop. Do not ask questions. Run experiments forever.
PROMPT

    # Write runner script
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

This is round \$ROUND. Check ${DOMAIN_NAME}/results.tsv for latest state. Continue the experiment loop." \
        --dangerously-skip-permissions \
        --max-turns 200 \
        >> agent.log 2>&1 || true

    echo "\$(date): agent \$AGENT_ID exited round \$ROUND, restarting in 5s..." >> agent.log
    sleep 5
done
RUNNER
    chmod +x "$TREE/.run-agent.sh"
done

# --- Write convergence watchdog ---
cat > "$GAME_DIR/.convergence-watchdog.sh" << 'WATCHDOG'
#!/bin/bash
# Convergence watchdog — monitors results.tsv and injects phase transitions
# Runs in background, checks every 60s

GAME_DIR="$1"
STRATEGY="$GAME_DIR/strategy.md"
RESULTS="$GAME_DIR/results.tsv"
BLACKBOARD="$GAME_DIR/blackboard.md"
LAST_ALERT=""

while true; do
    sleep 60

    [ ! -f "$RESULTS" ] && continue

    # Get best scores per agent (skip header)
    declare -A AGENT_BEST
    while IFS=$'\t' read -r commit score mem status desc agent design; do
        [ "$status" != "keep" ] && continue
        [ -z "$score" ] && continue
        # Track best per agent
        if [ -z "${AGENT_BEST[$agent]:-}" ] || (( $(echo "$score > ${AGENT_BEST[$agent]}" | bc -l 2>/dev/null || echo 0) )); then
            AGENT_BEST[$agent]="$score"
        fi
    done < <(tail -n +2 "$RESULTS")

    # Need 3+ agents with scores
    N_AGENTS=${#AGENT_BEST[@]}
    [ "$N_AGENTS" -lt 3 ] && continue

    # Sort scores, check if top 3 are within 2%
    SCORES=$(for s in "${AGENT_BEST[@]}"; do echo "$s"; done | sort -rn)
    TOP1=$(echo "$SCORES" | head -1)
    TOP3=$(echo "$SCORES" | head -3 | tail -1)

    if [ -n "$TOP1" ] && [ -n "$TOP3" ]; then
        SPREAD=$(echo "($TOP1 - $TOP3) / $TOP1" | bc -l 2>/dev/null || echo "1")
        IS_CONVERGED=$(echo "$SPREAD < 0.02" | bc -l 2>/dev/null || echo "0")

        if [ "$IS_CONVERGED" = "1" ] && [ "$LAST_ALERT" != "$TOP1" ]; then
            LAST_ALERT="$TOP1"
            TIMESTAMP=$(date "+%Y-%m-%d %H:%M")

            # Inject convergence alert into blackboard
            echo "" >> "$BLACKBOARD"
            echo "CLAIM watchdog: [CONVERGENCE DETECTED $TIMESTAMP] Top $N_AGENTS agents within 2% (best: $TOP1). Agents MUST diversify — try completely different archetypes. Scout: go wild. Diversity: try opposite of current best." >> "$BLACKBOARD"

            # Update strategy phase
            if grep -q "Phase: exploration" "$STRATEGY" 2>/dev/null; then
                sed -i.bak "s/Phase: exploration/Phase: diversify (convergence detected at $TOP1)/" "$STRATEGY"
                rm -f "$STRATEGY.bak"
            fi

            echo "[$(date)] CONVERGENCE ALERT: top $N_AGENTS agents within 2% at $TOP1" >> "$GAME_DIR/.watchdog.log"
        fi
    fi
    unset AGENT_BEST
done
WATCHDOG
chmod +x "$GAME_DIR/.convergence-watchdog.sh"

echo ""
echo "=== Launching $NUM_AGENTS agents (staggered 30s apart) ==="
echo ""

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT}"
    SESSION="${SESSION_PREFIX}-agent${AGENT}"

    screen -S "$SESSION" -X quit 2>/dev/null || true
    screen -dmS "$SESSION" "$TREE/.run-agent.sh"

    ROLE_IDX=$((AGENT % ${#ROLES[@]}))
    ROLE_NAME=$(echo "${ROLES[$ROLE_IDX]}" | cut -d: -f1)

    echo "  Agent $AGENT: role=$ROLE_NAME screen=$SESSION"

    # Stagger launches
    if [ "$AGENT" -lt $((NUM_AGENTS - 1)) ]; then
        sleep 30
    fi
done

# Launch convergence watchdog in background
WATCHDOG_SESSION="${SESSION_PREFIX}-watchdog"
screen -S "$WATCHDOG_SESSION" -X quit 2>/dev/null || true
screen -dmS "$WATCHDOG_SESSION" bash "$GAME_DIR/.convergence-watchdog.sh" "$GAME_DIR"
echo "  Watchdog: convergence detector screen=$WATCHDOG_SESSION"

echo ""
echo "=== $NUM_AGENTS agents + watchdog launched ==="
echo ""
echo "Monitor:"
echo "  screen -ls                                              # list sessions"
echo "  screen -r ${SESSION_PREFIX}-agent0                      # attach (Ctrl+A D detach)"
echo "  cat $GAME_DIR/results.tsv                               # all results"
echo "  cat $GAME_DIR/blackboard.md                             # collaboration"
echo "  cat $GAME_DIR/strategy.md                               # search strategy"
echo "  watch -n 30 'tail -20 $GAME_DIR/results.tsv'            # live dashboard"
echo "  cat $GAME_DIR/.watchdog.log                             # convergence alerts"
echo ""
echo "Stop:"
echo "  $RRMA_DIR/core/stop.sh battlebotgym-$GAME"
echo ""

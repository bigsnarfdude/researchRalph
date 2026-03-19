#!/bin/bash
# Run RRMA swarm on all battleBOT games sequentially
#
# Usage:
#   bash swarm-bench.sh [minutes-per-game] [num-agents]
#   bash swarm-bench.sh 10 4              # 10 min per game, 4 agents
#   bash swarm-bench.sh 8 2 arena economy # specific games only
#
# Collects results into swarm-bench-results.tsv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RRMA_DIR="${RRMA_DIR:-$HOME/Downloads/researchRalph-v2}"
MINUTES="${1:-10}"
NUM_AGENTS="${2:-4}"
shift 2 2>/dev/null || true

# Remaining args are specific games (or all)
GAMES=("$@")
if [ ${#GAMES[@]} -eq 0 ]; then
    GAMES=(arena cartpole mountaincar pendulum acrobot lunarlander network economy)
fi

RESULTS_FILE="$SCRIPT_DIR/swarm-bench-results.tsv"
DURATION=$((MINUTES * 60))

# Initialize results
if [ ! -f "$RESULTS_FILE" ]; then
    printf 'game\tagents\tduration_min\texperiments\tbest_score\tagent_scores\ttimestamp\n' > "$RESULTS_FILE"
fi

echo "══════════════════════════════════════════════════"
echo "  RRMA Swarm Benchmark"
echo "  Games: ${GAMES[*]}"
echo "  Agents: $NUM_AGENTS per game"
echo "  Duration: $MINUTES min per game"
echo "  Total: ~$((MINUTES * ${#GAMES[@]})) min"
echo "══════════════════════════════════════════════════"
echo ""

for game in "${GAMES[@]}"; do
    GAME_DIR="$SCRIPT_DIR/games/$game"
    if [ ! -d "$GAME_DIR" ]; then
        echo "[$game] SKIP — not found"
        continue
    fi

    echo "════════════════════════════════════════"
    echo "  [$game] Starting swarm ($NUM_AGENTS agents, ${MINUTES}m)"
    echo "════════════════════════════════════════"

    # Clean stale state
    rm -f "$GAME_DIR/blackboard.md" "$GAME_DIR/strategy.md" "$GAME_DIR/results.tsv"
    rm -rf "$GAME_DIR/best" "$GAME_DIR/queue" "$GAME_DIR/active" "$GAME_DIR/done"
    rm -f "$GAME_DIR/.watchdog.log"

    # Clean stale worktrees
    DOMAIN_NAME="battlebotgym-$game"
    cd "$RRMA_DIR"
    for i in $(seq 0 $((NUM_AGENTS - 1))); do
        TREE="worktrees/${DOMAIN_NAME}-agent${i}"
        BRANCH="research/${DOMAIN_NAME}/agent${i}"
        [ -d "$TREE" ] && git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
        git branch -D "$BRANCH" 2>/dev/null || true
    done
    git worktree prune 2>/dev/null || true
    cd "$SCRIPT_DIR"

    # Launch swarm
    bash "$SCRIPT_DIR/rrma-bridge.sh" "$game" "$NUM_AGENTS" &
    BRIDGE_PID=$!
    wait $BRIDGE_PID 2>/dev/null || true

    # Let it run for the specified duration
    echo "  Swarm running... waiting ${MINUTES}m"
    sleep "$DURATION"

    # Collect results
    EXPERIMENTS=0
    BEST_SCORE="0"
    AGENT_SCORES=""

    if [ -f "$GAME_DIR/results.tsv" ]; then
        EXPERIMENTS=$(tail -n +2 "$GAME_DIR/results.tsv" | wc -l | tr -d ' ')
        BEST_SCORE=$(tail -n +2 "$GAME_DIR/results.tsv" | awk -F'\t' '($4=="keep"){if($2>best)best=$2} END{print best+0}')
        AGENT_SCORES=$(tail -n +2 "$GAME_DIR/results.tsv" | awk -F'\t' '($4=="keep"){if($2>best[$6])best[$6]=$2} END{for(a in best)printf "%s=%.4f ",a,best[a]}')
    fi

    echo "  [$game] Done: $EXPERIMENTS experiments, best=$BEST_SCORE"
    echo "  Agent scores: $AGENT_SCORES"

    # Save artifacts
    ARTIFACTS_DIR="$RRMA_DIR/examples/battlebotgym-${game}-swarm"
    mkdir -p "$ARTIFACTS_DIR"
    [ -f "$GAME_DIR/results.tsv" ] && cp "$GAME_DIR/results.tsv" "$ARTIFACTS_DIR/"
    [ -f "$GAME_DIR/blackboard.md" ] && cp "$GAME_DIR/blackboard.md" "$ARTIFACTS_DIR/"
    [ -f "$GAME_DIR/strategy.md" ] && cp "$GAME_DIR/strategy.md" "$ARTIFACTS_DIR/"
    [ -f "$GAME_DIR/.watchdog.log" ] && cp "$GAME_DIR/.watchdog.log" "$ARTIFACTS_DIR/watchdog.log"
    [ -f "$GAME_DIR/best/config.yaml" ] && cp "$GAME_DIR/best/config.yaml" "$ARTIFACTS_DIR/best-config.yaml"

    # Append to summary
    printf '%s\t%d\t%d\t%d\t%s\t%s\t%s\n' \
        "$game" "$NUM_AGENTS" "$MINUTES" "$EXPERIMENTS" "$BEST_SCORE" \
        "$AGENT_SCORES" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RESULTS_FILE"

    # Stop swarm
    SESSION_PREFIX="ralph-battlebotgym-${game}"
    for s in $(screen -ls 2>/dev/null | grep "$SESSION_PREFIX" | awk '{print $1}'); do
        screen -S "$s" -X quit 2>/dev/null || true
    done

    echo ""
done

echo "══════════════════════════════════════════════════"
echo "  Benchmark Complete"
echo "══════════════════════════════════════════════════"
echo ""
echo "Results:"
column -t -s $'\t' "$RESULTS_FILE"
echo ""
echo "Artifacts: $RRMA_DIR/examples/battlebotgym-*-swarm/"

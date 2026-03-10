#!/bin/bash
# Health check and dashboard for running agents
#
# Usage:
#   ./core/monitor.sh <domain-dir>
#   watch -n 30 './core/monitor.sh domains/gpt2-tinystories'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/monitor.sh <domain-dir>}"
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
WORKTREE_DIR="$REPO_DIR/worktrees"

echo "=== researchRalph v2 — $DOMAIN_NAME ==="
echo "$(date)"
echo ""

# --- Agent Status ---
echo "--- AGENTS ---"
screen -ls 2>/dev/null | grep "ralph-${DOMAIN_NAME}" | while read -r line; do
    session=$(echo "$line" | awk '{print $1}')
    name=$(echo "$session" | cut -d. -f2-)
    echo "  RUNNING: $name"
done
echo ""

# --- Results Summary ---
echo "--- RESULTS ---"
if [ -f "$DOMAIN_DIR/results.tsv" ]; then
    total=$(tail -n +2 "$DOMAIN_DIR/results.tsv" | wc -l | tr -d ' ')
    keeps=$(tail -n +2 "$DOMAIN_DIR/results.tsv" | grep -c "keep" || echo 0)
    echo "  Total experiments: $total"
    echo "  Kept: $keeps ($(( total > 0 ? keeps * 100 / total : 0 ))%)"
    echo ""
    echo "  Last 5:"
    tail -5 "$DOMAIN_DIR/results.tsv" | while IFS=$'\t' read -r commit score mem status desc agent design; do
        printf "    %s  %-8s  %s  %s\n" "$score" "$status" "${agent:-?}" "${desc:0:50}"
    done
else
    echo "  No results yet."
fi
echo ""

# --- Blackboard ---
echo "--- BLACKBOARD (last 5 lines) ---"
if [ -f "$DOMAIN_DIR/blackboard.md" ]; then
    grep -E "^(CLAIM|RESPONSE|REQUEST|REFUTE) " "$DOMAIN_DIR/blackboard.md" 2>/dev/null | tail -5 | while read -r line; do
        echo "  ${line:0:80}"
    done
    count=$(grep -cE "^(CLAIM|RESPONSE|REQUEST|REFUTE) " "$DOMAIN_DIR/blackboard.md" 2>/dev/null || echo 0)
    echo "  ($count total messages)"
else
    echo "  No blackboard yet."
fi
echo ""

# --- Agent Health ---
echo "--- HEALTH ---"
for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
    [ -d "$tree" ] || continue
    agent=$(basename "$tree")
    if [ -f "$tree/run.log" ]; then
        last_mod=$(stat -f %m "$tree/run.log" 2>/dev/null || stat -c %Y "$tree/run.log" 2>/dev/null || echo 0)
        now=$(date +%s)
        age=$(( now - last_mod ))
        if [ "$age" -gt 600 ]; then
            echo "  WARNING: $agent — run.log stale (${age}s ago)"
        else
            echo "  OK: $agent — active (${age}s ago)"
        fi
    else
        echo "  UNKNOWN: $agent — no run.log"
    fi
done
echo ""

# --- Strategy ---
echo "--- STRATEGY (first 10 lines) ---"
if [ -f "$DOMAIN_DIR/strategy.md" ]; then
    head -10 "$DOMAIN_DIR/strategy.md" | sed 's/^/  /'
else
    echo "  No strategy yet."
fi

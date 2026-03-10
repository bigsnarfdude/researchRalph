#!/bin/bash
# Collect results from all agent worktrees into a single archive
#
# Usage:
#   ./core/collect.sh <domain-name> [output-dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_NAME="${1:?Usage: ./core/collect.sh <domain-name> [output-dir]}"
OUTPUT_DIR="${2:-$REPO_DIR/results/${DOMAIN_NAME}_$(date +%Y%m%d_%H%M%S)}"
WORKTREE_DIR="$REPO_DIR/worktrees"
DOMAIN_DIR="$REPO_DIR/domains/$DOMAIN_NAME"

mkdir -p "$OUTPUT_DIR"

echo "=== Collecting results for $DOMAIN_NAME ==="

# Copy shared state
for f in results.tsv blackboard.md strategy.md; do
    [ -f "$DOMAIN_DIR/$f" ] && cp "$DOMAIN_DIR/$f" "$OUTPUT_DIR/"
done

[ -d "$DOMAIN_DIR/best" ] && cp -r "$DOMAIN_DIR/best" "$OUTPUT_DIR/"
[ -d "$DOMAIN_DIR/done" ] && cp -r "$DOMAIN_DIR/done" "$OUTPUT_DIR/"

# Copy per-agent state
for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
    [ -d "$tree" ] || continue
    agent=$(basename "$tree")
    mkdir -p "$OUTPUT_DIR/$agent"

    [ -d "$tree/memory" ] && cp -r "$tree/memory" "$OUTPUT_DIR/$agent/"
    [ -d "$tree/scratch" ] && cp -r "$tree/scratch" "$OUTPUT_DIR/$agent/"
    [ -d "$tree/judge" ] && cp -r "$tree/judge" "$OUTPUT_DIR/$agent/"
    [ -d "$tree/supervisor" ] && cp -r "$tree/supervisor" "$OUTPUT_DIR/$agent/"
    [ -d "$tree/debate" ] && cp -r "$tree/debate" "$OUTPUT_DIR/$agent/"
    [ -f "$tree/progress.md" ] && cp "$tree/progress.md" "$OUTPUT_DIR/$agent/"
    [ -f "$tree/next_ideas.md" ] && cp "$tree/next_ideas.md" "$OUTPUT_DIR/$agent/"
    [ -f "$tree/agent.log" ] && tail -100 "$tree/agent.log" > "$OUTPUT_DIR/$agent/agent.log.tail"
done

echo "Results collected to: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR/"

# Cleanup worktrees
echo ""
read -p "Remove worktrees? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
        [ -d "$tree" ] || continue
        git -C "$REPO_DIR" worktree remove --force "$tree" 2>/dev/null || rm -rf "$tree"
        echo "  Removed: $(basename "$tree")"
    done
fi

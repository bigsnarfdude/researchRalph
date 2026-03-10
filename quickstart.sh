#!/bin/bash
# researchRalph v2 — Get running in 60 seconds
#
# Usage:
#   ./quickstart.sh                    # interactive setup
#   ./quickstart.sh my-domain          # create a new domain
#
# This creates a domain directory, opens the config for editing,
# and starts a single agent loop.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  researchRalph v2"
echo "  ─────────────────"
echo ""

# Check prerequisites
if ! command -v claude &>/dev/null; then
    echo "  ERROR: Claude Code CLI not found."
    echo "  Install: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi

if [ $# -ge 1 ]; then
    DOMAIN="$1"
else
    echo "  What are you optimizing? (one word, e.g., 'training', 'prompts', 'compiler')"
    read -p "  > " DOMAIN
fi

DOMAIN_DIR="$REPO_DIR/domains/$DOMAIN"

if [ -d "$DOMAIN_DIR" ] && [ -f "$DOMAIN_DIR/program.md" ]; then
    echo ""
    echo "  Domain '$DOMAIN' already exists."
    echo ""
else
    echo ""
    echo "  Creating domain: $DOMAIN"
    cp -r "$REPO_DIR/domains/template" "$DOMAIN_DIR"
    echo "  Created: domains/$DOMAIN/"
    echo ""
fi

echo "  Your domain has 3 files to configure:"
echo ""
echo "    1. domains/$DOMAIN/config.yaml   ← parameters agents will tune"
echo "    2. domains/$DOMAIN/run.sh        ← harness: run config → output score"
echo "    3. domains/$DOMAIN/program.md    ← instructions for the agent"
echo ""
echo "  Reference domains to copy from:"
echo "    domains/gpt2-tinystories/   (GPT-2 training, 186 experiments proven)"
echo "    domains/af-elicitation/     (prompt optimization)"
echo ""
echo "  ─────────────────"
echo ""
echo "  Once configured, run:"
echo ""
echo "    Single agent:    ./core/run-single.sh domains/$DOMAIN"
echo "    Multi-agent:     ./core/launch.sh domains/$DOMAIN 4"
echo "    With GPUs:       ./core/launch.sh domains/$DOMAIN 8 --gpu"
echo "    Monitor:         ./core/monitor.sh domains/$DOMAIN"
echo "    Stop:            ./core/stop.sh $DOMAIN"
echo ""

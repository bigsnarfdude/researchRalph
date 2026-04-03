#!/bin/bash
AGENT_ID=1
TREE_DIR="/Users/vincent/researchRalph/worktrees/nirenberg-1d-chaos-haiku-h1-control-agent1"
cd "$TREE_DIR"


ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    echo "$(date): agent $AGENT_ID starting round $ROUND" >> agent.log

    claude --model haiku -p "$(cat .agent-prompt.txt)

This is round $ROUND. Check nirenberg-1d-chaos-haiku-h1-control/results.tsv for latest state. Continue the experiment loop."         --dangerously-skip-permissions         --max-turns 200         >> agent.log 2>&1 || true

    echo "$(date): agent $AGENT_ID exited round $ROUND, restarting in 5s..." >> agent.log
    sleep 5
done

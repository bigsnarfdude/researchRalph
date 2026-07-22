#!/bin/bash
# mock-worker.sh — impersonates one agent for island preflight. No model calls.
#
# Does exactly what the harness tells a real worker to do, once:
#   1. write to its own island's blackboard (unique marker, grep-able later)
#   2. edit its workspace file
#   3. call the oracle exactly once, the way worker_prompt.md says to
#
# Usage: ISLAND=a AGENT=agent0 DOMAIN=/path/to/island ./mock-worker.sh

set -euo pipefail
ISLAND="${ISLAND:?}"; AGENT="${AGENT:?}"; DOMAIN="${DOMAIN:?}"
MARK="mock-${ISLAND}-${AGENT}"

echo "- [${MARK}] canned finding: no-op config tweak, score cited from run.sh output" >> "$DOMAIN/blackboard.md"

mkdir -p "$DOMAIN/workspace/$AGENT"
EDITABLE="$(grep '^editable:' "$DOMAIN/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE="${EDITABLE:-answer.txt}"
if [ -f "$DOMAIN/$EDITABLE" ]; then
    cp "$DOMAIN/$EDITABLE" "$DOMAIN/workspace/$AGENT/$EDITABLE"   # real domains: valid seed file
else
    printf 'canned answer from %s: the quick brown fox jumps over the lazy dog\n' "$MARK" > "$DOMAIN/workspace/$AGENT/$EDITABLE"
fi

CLAUDE_AGENT_ID="$AGENT" bash "$DOMAIN/run.sh" "$MARK" "mock experiment by $MARK"

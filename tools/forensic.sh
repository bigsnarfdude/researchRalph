#!/bin/bash
# forensic.sh — Quick forensic query via claude -p
#
# Usage:
#   bash tools/forensic.sh "which agent was most productive?"
#   bash tools/forensic.sh "what did agent3 struggle with?"
#   bash tools/forensic.sh  # interactive mode
#
# Prereq: python3 tools/rrma_traces.py domains/rrma-lean --logs data/rrma_lean_logs --out /tmp/rrma_traces_test.jsonl

TRACES="${TRACES_JSONL:-/tmp/rrma_traces_test.jsonl}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$TRACES" ]; then
    echo "No traces at $TRACES"
    echo "Run: python3 tools/rrma_traces.py domains/rrma-lean --logs data/rrma_lean_logs --out /tmp/rrma_traces_test.jsonl"
    exit 1
fi

# Build index once
INDEX=$(python3 "$SCRIPT_DIR/trace_forensics.py" "$TRACES" --index-only 2>/dev/null)

ask() {
    local question="$1"
    # Pull targeted evidence based on the question
    local evidence=""

    # Auto-detect agent references and pull their thinking
    for agent in agent0 agent1 agent2 agent3 agent4 agent5 agent6 agent7; do
        if echo "$question" | grep -qi "$agent"; then
            evidence="$evidence

## ${agent} thinking (last 10 blocks)
$(python3 -c "
from tools.trace_forensics import TraceStore
import json
store = TraceStore('$TRACES')
blocks = store.get_agent_thinking('$agent', 10)
for b in blocks:
    print(f'[step {b[\"step_index\"]}] {b[\"thinking\"][:500]}')
    if b['artifact_reads']: print(f'  reads: {b[\"artifact_reads\"]}')
    if b['artifact_writes']: print(f'  writes: {b[\"artifact_writes\"]}')
    print()
" 2>/dev/null)"
        fi
    done

    # Auto-search for keywords (skip common words)
    for word in $(echo "$question" | tr ' ' '\n' | grep -v -E '^(which|what|who|how|why|did|was|the|and|for|with|from|most|least|any|all|show|me|tell|about)$' | head -3); do
        if [ ${#word} -gt 3 ]; then
            local hits=$(python3 -c "
from tools.trace_forensics import TraceStore
import json
store = TraceStore('$TRACES')
results = store.search_traces('$word', 8)
for r in results:
    print(f'{r[\"agent_id\"]} step {r[\"step_index\"]} [{r[\"found_in\"]}]: {r[\"snippet\"][:200]}')
" 2>/dev/null)
            if [ -n "$hits" ]; then
                evidence="$evidence

## Search results for '$word'
$hits"
            fi
        fi
    done

    # Pull blackboard if question mentions it
    if echo "$question" | grep -qi "blackboard\|shared\|coordinat\|influence\|communic"; then
        evidence="$evidence

## Blackboard content
$(python3 -c "
from tools.trace_forensics import TraceStore
store = TraceStore('$TRACES')
bb = store.get_artifact('blackboard')
print(bb.get('content', '(not available)')[:3000])
" 2>/dev/null)"
    fi

    # Pull influences if question mentions them
    if echo "$question" | grep -qi "influence\|cross.agent\|coordinat\|collaborat\|spread\|shared"; then
        evidence="$evidence

## Cross-agent influences
$(python3 -c "
from tools.trace_forensics import TraceStore
store = TraceStore('$TRACES')
infs = store.get_influences()
for i in infs[:20]:
    src = store._agent_for_trace(i['source_trace_id'])
    tgt = store._agent_for_trace(i['target_trace_id'])
    print(f'{src} -> {tgt} via {i[\"artifact_id\"]} ({i[\"influence_type\"]})')
" 2>/dev/null)"
    fi

    # Assemble prompt
    local prompt="You are a forensic analyst for multi-agent AI research systems.
You have structured trace data from an RRMA (Recursive Research Multi-Agent) run on the rrma-lean domain (Lean 4 theorem proving on MiniF2F).

$INDEX
$evidence

QUESTION: $question

Answer with specific evidence: cite agent IDs, step numbers, exact quotes from thinking blocks. Follow influence chains. Flag anomalies. Be direct."

    echo "$prompt" | claude -p --model sonnet 2>/dev/null
}

if [ -n "$1" ]; then
    ask "$*"
else
    echo "=== RRMA Trace Forensics (claude -p) ==="
    echo "Traces: $TRACES"
    echo "Type a question, Ctrl+C to quit."
    echo ""
    while true; do
        printf "forensic> "
        read -r question || break
        [ -z "$question" ] && continue
        [ "$question" = "quit" ] && break
        echo ""
        ask "$question"
        echo ""
    done
fi

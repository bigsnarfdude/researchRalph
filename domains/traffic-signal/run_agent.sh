#!/bin/bash
# Run a single gemini agent on this domain with retry on 429.
# Usage: bash run_agent.sh <agent_id>
#
# Reads blackboard + program files, runs gemini -p --yolo with retries.

AGENT_ID="${1:-agent_0}"
DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_RETRIES=5
RETRY_DELAY=30
MODEL="gemini-2.5-flash"

PROMPT=$(cat <<EOF
You are a traffic signal optimization agent. Your AGENT_ID is ${AGENT_ID}.

Read program_static.md for immutable rules.
Read program.md for current guidance.
Read blackboard.md to see what other agents have already tested — do NOT repeat experiments already claimed.
Read results.tsv to see all scores so far.

Then run 3 experiments following the agent lifecycle exactly:
1. Write CLAIMED to blackboard.md before each run
2. Create workspace/${AGENT_ID}/ if it does not exist and copy best/config.yaml into it
3. Edit workspace/${AGENT_ID}/config.yaml with your chosen parameters
4. Run: bash run.sh <exp_name> "description" <design_type>
5. Replace CLAIMED with CLAIM on blackboard.md (always include avg_delay and evidence exp_id)
6. After all 3 experiments, append findings to LEARNINGS.md, MISTAKES.md, DESIRES.md

Pick experiments that have NOT been tested yet according to the blackboard.
Focus on what would be most informative given what is already known.
Never stop early. Never ask questions.
EOF
)

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[run_agent.sh] ${AGENT_ID} attempt ${attempt}/${MAX_RETRIES}"

    output=$(cd "$DOMAIN_DIR" && \
        CLAUDE_AGENT_ID="$AGENT_ID" \
        gemini -m "$MODEL" -p "$PROMPT" --yolo 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "429\|rateLimitExceeded\|MODEL_CAPACITY_EXHAUSTED"; then
        echo "[run_agent.sh] ${AGENT_ID} rate limited, waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        RETRY_DELAY=$((RETRY_DELAY * 2))
        continue
    fi

    echo "$output"
    echo "[run_agent.sh] ${AGENT_ID} done (exit ${exit_code})"
    exit 0
done

echo "[run_agent.sh] ${AGENT_ID} failed after ${MAX_RETRIES} attempts"
exit 1

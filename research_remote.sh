#!/bin/bash
# researchRalph with remote execution on nigel
# Usage: ./research_remote.sh [max_iterations]

set -e

MAX_ITERATIONS=${1:-20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/prompt_remote.md"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
HYPOTHESIS_FILE="$SCRIPT_DIR/hypothesis.json"
REMOTE_HOST="vincent@nigel.birs.ca"
REMOTE_DIR="~/researchRalph"

# Check SSH connectivity
echo "Checking SSH connection to nigel..."
if ! ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connected'" 2>/dev/null; then
    echo "ERROR: Cannot connect to $REMOTE_HOST"
    echo "Make sure SSH keys are set up and nigel is accessible."
    exit 1
fi
echo "SSH connection OK"

# Initialize progress file if needed
if [ ! -f "$PROGRESS_FILE" ]; then
    cat > "$PROGRESS_FILE" << 'EOF'
# researchRalph Progress Log

## Patterns (update this section with reusable insights)

- [None yet]

---

## Experiment Log

EOF
fi

echo ""
echo "Starting researchRalph (remote mode) - Max iterations: $MAX_ITERATIONS"
echo "Research question: $(jq -r '.research_question' $HYPOTHESIS_FILE)"
echo "Current best: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE) AUROC"
echo "Remote host: $REMOTE_HOST"
echo ""

for i in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  researchRalph Iteration $i of $MAX_ITERATIONS"
    echo "  $(date)"
    echo "═══════════════════════════════════════════════════════════"

    # Build full prompt with context
    FULL_PROMPT=$(cat "$PROMPT_FILE")
    FULL_PROMPT+="\n\n---\n\n## Current State\n\n"
    FULL_PROMPT+="### hypothesis.json\n\`\`\`json\n$(cat $HYPOTHESIS_FILE)\n\`\`\`\n\n"
    FULL_PROMPT+="### progress.txt (last 100 lines)\n\`\`\`\n$(tail -100 $PROGRESS_FILE)\n\`\`\`\n"

    # List existing probe files
    FULL_PROMPT+="\n\n### Existing probe files\n\`\`\`\n$(ls -1 probes/)\n\`\`\`\n"

    # List existing results
    if [ -d "results" ] && [ "$(ls -A results 2>/dev/null)" ]; then
        FULL_PROMPT+="\n\n### Existing results\n\`\`\`\n$(ls -1 results/)\n\`\`\`\n"
    fi

    # Run Claude CLI
    echo "Calling Claude..."
    OUTPUT=$(echo -e "$FULL_PROMPT" | claude --dangerously-skip-permissions 2>&1 | tee /dev/stderr) || true

    # Check for completion signals
    if echo "$OUTPUT" | grep -q "<promise>SUCCESS</promise>"; then
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  SUCCESS! Research goal achieved at iteration $i"
        echo "═══════════════════════════════════════════════════════════"
        echo "Best probe: $(jq -r '.current_best.probe_file' $HYPOTHESIS_FILE)"
        echo "Test AUROC: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE)"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "<promise>PLATEAU</promise>"; then
        echo ""
        echo "Research reached plateau - needs human guidance"
        echo "Check progress.txt for insights"
        exit 1
    fi

    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo ""
        echo "Research complete - search space exhausted"
        echo "Best achieved: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE) AUROC"
        exit 0
    fi

    # Check if a new probe was created and needs to be run
    # Look for the most recent probe file that doesn't have results yet
    for probe_file in probes/exp*.py; do
        if [ -f "$probe_file" ]; then
            exp_id=$(basename "$probe_file" .py)
            if [ ! -f "results/$exp_id.json" ]; then
                echo ""
                echo "Found new probe without results: $probe_file"
                echo "Running on nigel..."
                ./run_remote.sh "$probe_file"
            fi
        fi
    done

    echo ""
    echo "Iteration $i complete. Sleeping before next..."
    sleep 5
done

echo ""
echo "Reached max iterations ($MAX_ITERATIONS)"
echo "Current best: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE) AUROC"
exit 1

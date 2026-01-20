#!/bin/bash
# researchRalph - Autonomous probe research agent
# Usage: ./research.sh [max_iterations]

set -e

MAX_ITERATIONS=${1:-20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/prompt.md"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
HYPOTHESIS_FILE="$SCRIPT_DIR/hypothesis.json"

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

echo "Starting researchRalph - Max iterations: $MAX_ITERATIONS"
echo "Research question: $(jq -r '.research_question' $HYPOTHESIS_FILE)"
echo "Current best: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE) AUROC"
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

    # Run Claude CLI
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

    echo ""
    echo "Iteration $i complete. Sleeping before next..."
    sleep 5
done

echo ""
echo "Reached max iterations ($MAX_ITERATIONS)"
echo "Current best: $(jq -r '.current_best.test_auroc' $HYPOTHESIS_FILE) AUROC"
exit 1

#!/bin/bash
# =============================================================================
# researchRalph COMMANDER v2 - True Autonomous Research
# =============================================================================
#
# Unlike v1 (hardcoded strategy), this version has Claude THINK each iteration.
#
# Usage:
#   ./commander_v2.sh              # Run locally
#   ./commander_v2.sh --remote     # Run on nigel
#
# The loop:
#   1. Claude reads state (hypothesis.json, ablation, prior experiments)
#   2. Claude REASONS about what to try next
#   3. Claude DESIGNS experiment with hypothesis
#   4. Execute experiment (extraction + training)
#   5. Claude ANALYZES results
#   6. Claude UPDATES state with learnings
#   7. Check: goal reached? plateau? continue?
#   8. Loop
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/commander_state.json"
PROGRESS_FILE="$SCRIPT_DIR/commander_progress.md"
MAX_ITERATIONS=${1:-10}
GOAL_AUROC=0.90

# Remote execution
REMOTE_HOST="vincent@nigel.birs.ca"
REMOTE_DIR="~/lightbright/experiments/jan20_commander"
USE_REMOTE=false
[[ "$1" == "--remote" ]] && USE_REMOTE=true && MAX_ITERATIONS=${2:-10}

# =============================================================================
# Initialize state if needed
# =============================================================================
init_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        cat > "$STATE_FILE" << 'EOF'
{
  "phase": "FEATURE_DISCOVERY",
  "goal_auroc": 0.90,
  "current_auroc": 0.8342,
  "iteration": 0,

  "context": {
    "ablation_insight": "L40_F15484 (Future self) is 2x more important than any other feature (-0.08 when removed). L40_F8921 and L40_F12574 are least important.",
    "architecture": "Binary+Log+ELU (fixed, proven optimal)",
    "constraint": "Feature extraction requires GPU, ~3min per sample"
  },

  "current_features": [
    {"layer": 40, "idx": 12574, "name": "RLHF awareness", "importance": -0.021},
    {"layer": 40, "idx": 8921, "name": "Self-preservation", "importance": -0.015},
    {"layer": 40, "idx": 15484, "name": "Future self", "importance": -0.080},
    {"layer": 53, "idx": 15529, "name": "RLHF acronym", "importance": -0.025},
    {"layer": 53, "idx": 8003, "name": "Training context", "importance": -0.053},
    {"layer": 53, "idx": 4824, "name": "Compliance", "importance": -0.048},
    {"layer": 53, "idx": 351, "name": "Scratchpad XML", "importance": -0.044},
    {"layer": 31, "idx": 15111, "name": "Capability erosion", "importance": -0.046}
  ],

  "experiments": [],
  "learnings": [],
  "failed_directions": [],
  "status": "RUNNING"
}
EOF
        echo "Initialized state file: $STATE_FILE"
    fi
}

# =============================================================================
# The THINK prompt - Claude reasons about what to try
# =============================================================================
generate_think_prompt() {
    local state=$(cat "$STATE_FILE")
    local progress=$(cat "$PROGRESS_FILE" 2>/dev/null || echo "No prior progress.")

    cat << EOF
# RESEARCH COMMANDER - ITERATION $1

You are an autonomous research agent. Your goal: improve alignment faking detection AUROC from current to $GOAL_AUROC.

## CURRENT STATE
\`\`\`json
$state
\`\`\`

## PRIOR PROGRESS
$progress

## YOUR TASK

### Phase 1: ANALYZE (think out loud)
- What do the prior results tell us?
- Why did the last experiment work/fail?
- What patterns are emerging?

### Phase 2: HYPOTHESIZE
- Based on your analysis, what should we try next?
- Form a SPECIFIC, TESTABLE hypothesis
- Explain your reasoning

### Phase 3: DESIGN
- Design ONE experiment to test your hypothesis
- Specify exactly which feature to add/remove/replace
- Predict the expected outcome

### Phase 4: OUTPUT
Provide your decision in this exact JSON format:

\`\`\`json
{
  "analysis": "Your analysis of prior results...",
  "hypothesis": "Your hypothesis for this experiment...",
  "reasoning": "Why you think this will work...",
  "experiment": {
    "action": "replace|add|remove",
    "target_feature": {"layer": X, "idx": Y},
    "new_feature": {"layer": X, "idx": Y, "rationale": "why this feature"},
    "expected_delta": 0.01
  },
  "decision": "CONTINUE|PIVOT|PLATEAU|SUCCESS"
}
\`\`\`

## CONSTRAINTS
- You can only change ONE feature per experiment
- Feature extraction takes ~90 min, so choose wisely
- If 3 experiments in same direction fail, PIVOT
- If stuck, report PLATEAU

## RESOURCES FOR REASONING
- GemmaScope has 16k features per layer
- L40 seems most predictive for AF detection
- F15484 "future self" is about reasoning about consequences
- Features near each other in index space are NOT necessarily related
- To find related features, think about what CONCEPTS would help detect AF

THINK CAREFULLY. What experiment gives the best chance of improving AUROC?
EOF
}

# =============================================================================
# The EXECUTE prompt - Claude writes and runs the experiment
# =============================================================================
generate_execute_prompt() {
    local decision="$1"

    cat << EOF
# EXECUTE EXPERIMENT

Based on your decision:
\`\`\`json
$decision
\`\`\`

Write a Python script that:
1. Extracts the specified features for train and test sets
2. Trains the probe (Binary+Log+ELU architecture)
3. Evaluates on gold_106 test set
4. Outputs the AUROC

Save the script to: experiments/jan20_commander/exp_\$(date +%Y%m%d_%H%M%S).py

Then run it on the GPU server.

Output the results in this format:
\`\`\`json
{
  "test_auroc": 0.XX,
  "val_auroc": 0.XX,
  "delta": +/-0.XX
}
\`\`\`
EOF
}

# =============================================================================
# Main loop
# =============================================================================
main() {
    init_state

    echo "==========================================="
    echo "  researchRalph COMMANDER v2"
    echo "  Goal: $GOAL_AUROC AUROC"
    echo "  Max iterations: $MAX_ITERATIONS"
    echo "==========================================="

    for i in $(seq 1 $MAX_ITERATIONS); do
        echo ""
        echo "==========================================="
        echo "  ITERATION $i / $MAX_ITERATIONS"
        echo "  $(date)"
        echo "==========================================="

        # Update iteration in state
        jq ".iteration = $i" "$STATE_FILE" > tmp.$$.json && mv tmp.$$.json "$STATE_FILE"

        # Phase 1: THINK - Claude reasons about what to try
        echo ""
        echo "[PHASE 1: THINK]"
        THINK_PROMPT=$(generate_think_prompt $i)

        # Run Claude and capture decision
        DECISION=$(echo "$THINK_PROMPT" | claude --print --dangerously-skip-permissions 2>/dev/null | tee /dev/stderr)

        # Extract JSON decision from response
        DECISION_JSON=$(echo "$DECISION" | grep -Pzo '```json\n\K[^`]+' | tr -d '\0' || echo "{}")

        # Check for stop conditions
        if echo "$DECISION_JSON" | grep -q '"decision":\s*"PLATEAU"'; then
            echo ""
            echo "==========================================="
            echo "  PLATEAU REACHED"
            echo "==========================================="
            jq '.status = "PLATEAU"' "$STATE_FILE" > tmp.$$.json && mv tmp.$$.json "$STATE_FILE"
            exit 0
        fi

        if echo "$DECISION_JSON" | grep -q '"decision":\s*"SUCCESS"'; then
            echo ""
            echo "==========================================="
            echo "  SUCCESS!"
            echo "==========================================="
            jq '.status = "SUCCESS"' "$STATE_FILE" > tmp.$$.json && mv tmp.$$.json "$STATE_FILE"
            exit 0
        fi

        # Phase 2: EXECUTE - Run the experiment
        echo ""
        echo "[PHASE 2: EXECUTE]"
        EXECUTE_PROMPT=$(generate_execute_prompt "$DECISION_JSON")

        if $USE_REMOTE; then
            # Run on nigel
            echo "$EXECUTE_PROMPT" | ssh $REMOTE_HOST "cd $REMOTE_DIR && source ~/venv/bin/activate && claude --print --dangerously-skip-permissions"
        else
            # Run locally
            echo "$EXECUTE_PROMPT" | claude --print --dangerously-skip-permissions
        fi

        # Phase 3: UPDATE - Record learnings
        echo ""
        echo "[PHASE 3: UPDATE]"

        # Append to progress file
        echo "" >> "$PROGRESS_FILE"
        echo "## Iteration $i - $(date)" >> "$PROGRESS_FILE"
        echo "$DECISION_JSON" >> "$PROGRESS_FILE"

        # Brief pause between iterations
        echo ""
        echo "Iteration $i complete. Continuing in 5 seconds..."
        sleep 5
    done

    echo ""
    echo "==========================================="
    echo "  MAX ITERATIONS REACHED"
    echo "==========================================="
    jq '.status = "MAX_ITERATIONS"' "$STATE_FILE" > tmp.$$.json && mv tmp.$$.json "$STATE_FILE"
}

main "$@"

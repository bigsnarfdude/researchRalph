#!/bin/bash
# new_experiment.sh — Create a clean experiment domain for one matrix cell
#
# Usage: bash tools/new_experiment.sh <template-domain> <model> <chaos-ratio> <n-agents> [date]
#
# Examples:
#   bash tools/new_experiment.sh domains/nirenberg-1d-blind-chaos gemma4 0 2
#   bash tools/new_experiment.sh domains/nirenberg-1d-blind-chaos gemma4 50 4
#   bash tools/new_experiment.sh domains/pushover gemma4 37 8
#
# Creates: domains/<template>-<model>-c<chaos>-n<agents>-<date>/
# Everything reset to zero. No history, no blackboard seeds, no results.

set -euo pipefail

TEMPLATE="${1:?Usage: new_experiment.sh <template-domain> <model> <chaos-ratio> <n-agents> [date]}"
MODEL="${2:?Specify model name (e.g. gemma4, haiku, sonnet)}"
CHAOS_PCT="${3:?Specify chaos ratio 0-100 (e.g. 0, 25, 50)}"
N_AGENTS="${4:?Specify number of agents}"
DATE="${5:-$(date +%Y%m%d)}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/$TEMPLATE"

if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "ERROR: template domain not found: $TEMPLATE_DIR"
    exit 1
fi

# Derive base name from template
TEMPLATE_BASE="$(basename "$TEMPLATE")"
NEW_NAME="${TEMPLATE_BASE}-${MODEL}-c${CHAOS_PCT}-n${N_AGENTS}-${DATE}"
NEW_DIR="$REPO_ROOT/domains/$NEW_NAME"

if [ -d "$NEW_DIR" ]; then
    echo "ERROR: domain already exists: $NEW_DIR"
    echo "Delete it first or use a different date suffix."
    exit 1
fi

echo "=== new_experiment.sh ==="
echo "Template:  $TEMPLATE_DIR"
echo "New domain: $NEW_DIR"
echo "Model:     $MODEL"
echo "Chaos:     ${CHAOS_PCT}%"
echo "Agents:    $N_AGENTS"
echo ""

# Copy template — only the static scaffolding files
mkdir -p "$NEW_DIR"

for f in run.sh solve.py program.md program_static.md config.yaml chaos_prompt.md calibration.md MANIFEST.md; do
    [ -f "$TEMPLATE_DIR/$f" ] && cp "$TEMPLATE_DIR/$f" "$NEW_DIR/$f"
done

# Copy best/ (starting point for configs)
if [ -d "$TEMPLATE_DIR/best" ]; then
    cp -r "$TEMPLATE_DIR/best" "$NEW_DIR/best"
fi

# Fresh blackboard — just the header
DOMAIN_TITLE="$NEW_NAME"
cat > "$NEW_DIR/blackboard.md" << EOF
# Blackboard — ${DOMAIN_TITLE}

Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.
EOF

# Fresh empty results
echo -e "exp_id\tscore\tnorm\tstatus\tdescription\tagent\tdesign\telapsed_s\tsolution_energy\twall_time" > "$NEW_DIR/results.tsv"

# Experiment manifest
cat > "$NEW_DIR/EXPERIMENT.md" << EOF
# Experiment: ${DOMAIN_TITLE}

| Field | Value |
|-------|-------|
| Template | ${TEMPLATE_BASE} |
| Model | ${MODEL} |
| Chaos ratio | ${CHAOS_PCT}% |
| N agents | ${N_AGENTS} |
| Date | ${DATE} |
| Created | $(date) |

## Notes

EOF

echo "Created: $NEW_DIR"
echo ""
echo "Contents:"
ls "$NEW_DIR"
echo ""
echo "Launch:"
echo "  GEMINI_API_KEY=\$GEMINI_API_KEY \\"
echo "  CHAOS_OLLAMA_MODEL=gemma4:26b \\"
echo "  bash v4/launch-agents-chaos-v2.sh $NEW_DIR $N_AGENTS \"<chaos-ids>\" 30 5"

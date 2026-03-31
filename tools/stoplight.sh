#!/bin/bash
# Stoplight report: sync from nigel + run stoplight check
# Usage: bash tools/stoplight.sh domains/gpt2-tinystories-v44
#        bash tools/stoplight.sh domains/gpt2-tinystories-v44 --json

DOMAIN="${1:?Usage: stoplight.sh <domain> [--json]}"
shift
EXTRA_ARGS="$@"

REMOTE="vincent@nigel.birs.ca"
REMOTE_BASE="~/researchRalph"

# Files to sync
FILES=(
    results.tsv
    blackboard.md
    DESIRES.md
    MISTAKES.md
    LEARNINGS.md
    program.md
)

echo "Syncing ${DOMAIN} from nigel..."
for f in "${FILES[@]}"; do
    scp -q "${REMOTE}:${REMOTE_BASE}/${DOMAIN}/${f}" "${DOMAIN}/${f}" 2>/dev/null
done
echo ""

python3 tools/trustloop_stoplight.py "${DOMAIN}" ${EXTRA_ARGS}

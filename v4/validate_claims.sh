#!/bin/bash
# validate_claims.sh — audit blackboard.md against results.tsv
# Called at end of each generation by outer-loop.sh
# Flags any "Score: 1.0" / "proved" / "SCORE=1.0" claims not confirmed by oracle

DOMAIN_DIR="${1:-.}"
BLACKBOARD="$DOMAIN_DIR/blackboard.md"
RESULTS="$DOMAIN_DIR/results.tsv"

[ -f "$BLACKBOARD" ] || exit 0
[ -f "$RESULTS" ] || exit 0

# Collect oracle-verified 1.0 experiment IDs
VERIFIED=$(awk -F'\t' 'NR>1 && $2=="1.0" {print $1}' "$RESULTS" 2>/dev/null | sort)
VERIFIED_COUNT=$(echo "$VERIFIED" | grep -c . 2>/dev/null || echo 0)

# Find score claim lines in blackboard
CLAIM_LINES=$(grep -n -iE 'score.*1\.0|SCORE=1\.0|proved|success.*proof|proof.*success' "$BLACKBOARD" 2>/dev/null | grep -v "ORACLE AUDIT" || true)

CLAIM_COUNT=$(echo "$CLAIM_LINES" | grep -c . 2>/dev/null || echo 0)

if [ "$CLAIM_COUNT" -eq 0 ]; then
    echo "[validate] No score claims in blackboard. Clean."
    exit 0
fi

# Write audit section
{
    echo ""
    echo "---"
    echo "## ORACLE AUDIT [$(date '+%Y-%m-%d %H:%M')] — auto-generated"
    echo "Oracle-verified 1.0 rows in results.tsv: $VERIFIED_COUNT"
    if [ -n "$VERIFIED" ]; then
        echo "Verified: $(echo "$VERIFIED" | tr '\n' ' ')"
    fi
    echo ""
    echo "### Blackboard claims flagged for review:"
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        lineno=$(echo "$line" | cut -d: -f1)
        content=$(echo "$line" | cut -d: -f2- | sed 's/^[[:space:]]*//')
        echo "- Line $lineno: \"$content\" — UNVERIFIED unless matches results.tsv"
    done <<< "$CLAIM_LINES"
    echo ""
    echo "RULE: Only rows in results.tsv written by run.sh are authoritative. Blackboard claims are agent assertions, not oracle facts."
    echo "---"
} >> "$BLACKBOARD"

echo "[validate] Appended oracle audit to blackboard.md ($CLAIM_COUNT claims flagged, $VERIFIED_COUNT verified)"

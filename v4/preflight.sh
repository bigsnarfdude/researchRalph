#!/bin/bash
# preflight.sh — automated RRMA deployment checklist
#
# Codifies the manual checks from CLAUDE.md that would have caught the
# erdos-125 burn (~$14, 300+ agent-turns, 0 experiments logged):
#   1. Oracle reads the agent workspace file, not the domain root
#   2. Oracle logs a row to results.tsv
#   3. A worker workflow template exists for this domain type
#
# The live oracle test (checks 1+2) runs only for lean_proof domains — an ML
# run.sh is a full training run. ML domains get static checks instead.
#
# Usage: bash preflight.sh /path/to/domain
# Exit 0 = safe to launch. Exit 1 = do not launch.

DOMAIN_DIR="${1:-.}"
DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FAILED=0
fail() { echo "[preflight] FAIL: $1"; FAILED=1; }
ok()   { echo "[preflight] ok: $1"; }

DOMAIN_TYPE="$(grep '^domain_type:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE_FILE="$(grep '^editable:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
EDITABLE_FILE="${EDITABLE_FILE:-train.py}"

echo "[preflight] domain: $(basename "$DOMAIN_DIR") (type: ${DOMAIN_TYPE:-unset}, editable: $EDITABLE_FILE)"

# --- Check 3: worker workflow template exists for this domain type ---
if [ -f "$DOMAIN_DIR/worker_prompt.md" ]; then
    ok "worker workflow: domain-local worker_prompt.md"
elif [ -n "$DOMAIN_TYPE" ] && [ -f "$SCRIPT_DIR/prompts/$DOMAIN_TYPE.md" ]; then
    ok "worker workflow: v4/prompts/$DOMAIN_TYPE.md"
elif [ -f "$SCRIPT_DIR/prompts/ml_default.md" ]; then
    echo "[preflight] WARN: no template for domain_type='${DOMAIN_TYPE:-unset}' — launch will fall back to ml_default.md. Verify that workflow matches this domain."
else
    fail "no worker workflow template found (domain worker_prompt.md or v4/prompts/*.md)"
fi

# --- Static checks on run.sh ---
RUN_SH="$DOMAIN_DIR/run.sh"
if [ ! -f "$RUN_SH" ]; then
    fail "run.sh missing"
else
    if grep -qE 'CLAUDE_AGENT_ID|AGENT_ID' "$RUN_SH"; then
        ok "run.sh reads CLAUDE_AGENT_ID"
    else
        fail "run.sh never reads CLAUDE_AGENT_ID — it cannot find agent workspaces"
    fi
    if grep -q 'workspace' "$RUN_SH"; then
        ok "run.sh references workspace/"
    else
        fail "run.sh never references workspace/ — agents' edits will be invisible to the oracle"
    fi
    if grep -q 'results.tsv' "$RUN_SH"; then
        ok "run.sh writes results.tsv"
    else
        fail "run.sh never touches results.tsv — experiments will not be logged"
    fi
fi

# --- Live oracle test (lean_proof only) ---
if [ "$DOMAIN_TYPE" = "lean_proof" ] && [ "$FAILED" -eq 0 ]; then
    echo "[preflight] running live oracle test as agent0..."
    WS="$DOMAIN_DIR/workspace/agent0"
    mkdir -p "$WS"
    if [ ! -f "$WS/$EDITABLE_FILE" ]; then
        if [ -f "$DOMAIN_DIR/best/$EDITABLE_FILE" ]; then
            cp "$DOMAIN_DIR/best/$EDITABLE_FILE" "$WS/$EDITABLE_FILE"
        elif [ -f "$DOMAIN_DIR/$EDITABLE_FILE" ]; then
            cp "$DOMAIN_DIR/$EDITABLE_FILE" "$WS/$EDITABLE_FILE"
        fi
    fi

    ROWS_BEFORE=$(awk 'NR>1' "$DOMAIN_DIR/results.tsv" 2>/dev/null | wc -l | tr -d ' ')
    ORACLE_OUT=$(cd "$DOMAIN_DIR" && CLAUDE_AGENT_ID=agent0 bash run.sh 2>&1)
    echo "$ORACLE_OUT" | sed 's/^/[preflight oracle] /'

    if echo "$ORACLE_OUT" | grep -q "workspace/agent0"; then
        ok "oracle reads workspace/agent0 (not the domain root file)"
    else
        fail "oracle output never mentions workspace/agent0 — it is reading the wrong file"
    fi

    ROWS_AFTER=$(awk 'NR>1' "$DOMAIN_DIR/results.tsv" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$ROWS_AFTER" -gt "$ROWS_BEFORE" ]; then
        ok "oracle logged a row to results.tsv"
        # Remove the test row so it doesn't pollute results (preserves file mode)
        python3 - "$DOMAIN_DIR/results.tsv" <<'PY'
import os, stat, sys
p = sys.argv[1]
mode = stat.S_IMODE(os.stat(p).st_mode)
os.chmod(p, 0o644)
with open(p) as f:
    lines = f.readlines()
with open(p, "w") as f:
    f.writelines(lines[:-1])
os.chmod(p, mode)
PY
        echo "[preflight] removed test row from results.tsv"
    else
        fail "oracle ran but results.tsv gained no row — logging is broken"
    fi
fi

if [ "$FAILED" -ne 0 ]; then
    echo "[preflight] RESULT: FAIL — do not launch agents until the above is fixed."
    exit 1
fi
echo "[preflight] RESULT: PASS"

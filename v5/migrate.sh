#!/bin/bash
# migrate.sh — inject a distilled digest from one island's board into the others.
#
# Distilled findings only, never raw boards (raw-board migration re-correlates
# the islands and defeats the whole design). The digest is authored by the
# advisor; with ADVISOR_STUB set it is read from a canned file instead, which
# is how the preflight suite and future cheap-worker smoke tests run.
#
# Idempotent per (source, target): a digest marker already present on the
# target skips the append, so a re-run never spams boards.
# Never writes to the source island. Charges the target's line budget after.
#
# Usage: [ADVISOR_STUB=digest.md] bash migrate.sh --from /path/isl-a --to /path/isl-b /path/isl-c

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FROM=""; TARGETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to)   shift; while [ $# -gt 0 ]; do TARGETS+=("$1"); shift; done ;;
        *) echo "ERROR: unknown arg $1" >&2; exit 1 ;;
    esac
done
[ -n "$FROM" ] && [ "${#TARGETS[@]}" -gt 0 ] || { echo "usage: migrate.sh --from <island> --to <island>..." >&2; exit 1; }
FROM="$(cd "$FROM" && pwd)"
SRC="$(basename "$FROM")"
MARKER="<!-- DIGEST(from=$SRC) -->"

# --- Author the digest ---
if [ -n "${ADVISOR_STUB:-}" ]; then
    [ -f "$ADVISOR_STUB" ] || { echo "ERROR: ADVISOR_STUB=$ADVISOR_STUB not found" >&2; exit 1; }
    DIGEST="$(cat "$ADVISOR_STUB")"
else
    # Live advisor call lands in v5.1 (outer-loop NUDGE wiring). Failing loud
    # here is deliberate — silent no-op migration is the worst failure mode.
    echo "ERROR: live advisor not wired yet (v5.1) — set ADVISOR_STUB=<digest file>" >&2
    exit 3
fi

for T in "${TARGETS[@]}"; do
    T="$(cd "$T" && pwd)"
    if [ "$T" = "$FROM" ]; then
        echo "[migrate] skip: refusing self-migration into $SRC"
        continue
    fi
    BOARD="$T/blackboard.md"
    if grep -qF "$MARKER" "$BOARD"; then
        echo "[migrate] skip: $(basename "$T") already has digest from $SRC"
        continue
    fi
    { printf '\n%s\n' "$MARKER"; printf '%s\n' "$DIGEST"; } >> "$BOARD"
    echo "[migrate] digest from $SRC -> $(basename "$T")"
    bash "$SCRIPT_DIR/board-budget.sh" "$T" || true   # charge the budget; over-cap is a flag, not a rollback
done

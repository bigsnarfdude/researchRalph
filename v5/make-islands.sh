#!/bin/bash
# make-islands.sh — clone a domain into K sibling island domains.
#
# Islands are full domain clones (<domain>-isl-a, -isl-b, ...), reusing the
# existing per-domain isolation: launcher, worker prompts, diagnose.py and
# preflight.sh all work unchanged on an island because an island IS a domain.
# Each island gets a fresh blackboard.md, fresh results.tsv (header only),
# and an empty workspace/.
#
# Usage: bash make-islands.sh /path/to/domain [K] [--force]
#   K defaults to 3. --force wipes existing island dirs (clean fixture reset).

set -euo pipefail

DOMAIN_DIR="${1:?usage: make-islands.sh /path/to/domain [K] [--force]}"
DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
K="${2:-3}"
FORCE=0
for arg in "$@"; do [ "$arg" = "--force" ] && FORCE=1; done

SUFFIXES=(a b c d e f g h)
BASE_NAME="$(basename "$DOMAIN_DIR")"
PARENT="$(dirname "$DOMAIN_DIR")"

# Fresh results.tsv keeps the base domain's header row if it has one
HEADER="$(head -n 1 "$DOMAIN_DIR/results.tsv" 2>/dev/null || true)"
[ -n "$HEADER" ] || HEADER=$'exp_id\tscore\ttime\tstatus\tdescription\tagent\tdesign'

for i in $(seq 0 $((K-1))); do
    ISL="$PARENT/$BASE_NAME-isl-${SUFFIXES[$i]}"
    if [ -d "$ISL" ]; then
        if [ "$FORCE" -eq 1 ]; then
            rm -rf "$ISL"
        else
            echo "ERROR: $ISL exists (use --force to reset)" >&2
            exit 1
        fi
    fi
    mkdir -p "$ISL/workspace"
    # Copy the domain machinery; boards/results/workspaces start fresh.
    # The editable: file from config.yaml rides along — the launcher seeds
    # agent workspaces from it.
    EDITABLE="$(grep '^editable:' "$DOMAIN_DIR/config.yaml" 2>/dev/null | awk '{print $2}')"
    for f in run.sh config.yaml program.md worker_prompt.md engine.py env.yaml train_config.yaml $EDITABLE; do
        [ -f "$DOMAIN_DIR/$f" ] && cp "$DOMAIN_DIR/$f" "$ISL/$f"
    done
    printf '# Blackboard — %s (island %s)\n' "$BASE_NAME" "${SUFFIXES[$i]}" > "$ISL/blackboard.md"
    printf '%s\n' "$HEADER" > "$ISL/results.tsv"
    echo "[make-islands] created $ISL"
done

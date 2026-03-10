#!/bin/bash
# researchRalph v2 — Reactive dispatch from blackboard REQUESTs
#
# Watches the blackboard for REQUEST lines and spawns ephemeral agents.
#
# Usage:
#   ./core/conductor.sh <domain-dir>
#   ./core/conductor.sh domains/gpt2-tinystories --dry-run
#   ./core/conductor.sh domains/af-elicitation --max 8 --poll 15

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/conductor.sh <domain-dir> [--dry-run] [--max N] [--poll N]}"
shift

# Resolve to absolute path
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
POLL_INTERVAL=30
MAX_CONCURRENT=4
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --poll) POLL_INTERVAL="$2"; shift 2 ;;
        --max) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BLACKBOARD="$DOMAIN_DIR/blackboard.md"
CONDUCTOR_DIR="$DOMAIN_DIR/.conductor"
WORKTREE_DIR="$REPO_DIR/worktrees"

mkdir -p "$CONDUCTOR_DIR/dispatched" "$CONDUCTOR_DIR/log"
LOG="$CONDUCTOR_DIR/log/conductor_$(date +%Y%m%d_%H%M%S).log"

log() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

get_pending_requests() {
    [ -f "$BLACKBOARD" ] || return
    grep -n "^REQUEST " "$BLACKBOARD" 2>/dev/null | while IFS= read -r line; do
        content=$(echo "$line" | cut -d: -f2-)
        hash=$(echo "$content" | md5 -q 2>/dev/null || echo "$content" | md5sum | cut -d' ' -f1)
        if ! [ -f "$CONDUCTOR_DIR/dispatched/$hash" ]; then
            line_num=$(echo "$line" | cut -d: -f1)
            echo "$hash|$line_num|$content"
        fi
    done
}

count_running() {
    screen -ls 2>/dev/null | grep -c "conductor-${DOMAIN_NAME}" || echo 0
}

dispatch_request() {
    local hash="$1" line_num="$2" content="$3"

    if $DRY_RUN; then
        log "[DRY RUN] Would dispatch: $content"
        return
    fi

    echo "$(date +%Y%m%d_%H%M%S)" > "$CONDUCTOR_DIR/dispatched/$hash"

    local branch="conductor/${DOMAIN_NAME}/${hash:0:8}"
    local tree="$WORKTREE_DIR/conductor-${DOMAIN_NAME}-${hash:0:8}"

    git -C "$REPO_DIR" branch -D "$branch" 2>/dev/null || true
    [ -d "$tree" ] && { git -C "$REPO_DIR" worktree remove --force "$tree" 2>/dev/null || rm -rf "$tree"; }
    git -C "$REPO_DIR" worktree add -b "$branch" "$tree" HEAD

    rm -rf "$tree/$DOMAIN_NAME" && ln -sfn "$DOMAIN_DIR" "$tree/$DOMAIN_NAME"

    cat > "$tree/.agent-prompt.txt" << PROMPT
You are an EPHEMERAL agent spawned by the conductor to handle this request:

$content

Read $DOMAIN_NAME/program.md for context.
Read $DOMAIN_NAME/blackboard.md and $DOMAIN_NAME/results.tsv for state.

PROTOCOL:
1. Execute the requested experiment or analysis
2. Record results in $DOMAIN_NAME/results.tsv
3. Post findings to $DOMAIN_NAME/blackboard.md as CLAIM or RESPONSE
4. Exit when complete (do NOT loop forever)
PROMPT

    cat > "$tree/.run-conductor-agent.sh" << RUNNER
#!/bin/bash
cd "$tree"
echo "\$(date): conductor agent ${hash:0:8} starting" >> agent.log
claude -p "\$(cat .agent-prompt.txt)" \
    --dangerously-skip-permissions \
    --max-turns 50 \
    2>> agent.log || true
echo "\$(date): conductor agent ${hash:0:8} finished" >> agent.log
RUNNER
    chmod +x "$tree/.run-conductor-agent.sh"

    local session="conductor-${DOMAIN_NAME}-${hash:0:8}"
    screen -dmS "$session" "$tree/.run-conductor-agent.sh"
    log "DISPATCHED: $content → $session"
}

cleanup_finished() {
    for done_file in "$CONDUCTOR_DIR"/dispatched/*; do
        [ -f "$done_file" ] || continue
        local hash=$(basename "$done_file")
        local session="conductor-${DOMAIN_NAME}-${hash:0:8}"
        if ! screen -ls 2>/dev/null | grep -q "$session"; then
            local tree="$WORKTREE_DIR/conductor-${DOMAIN_NAME}-${hash:0:8}"
            [ -d "$tree" ] && git -C "$REPO_DIR" worktree remove --force "$tree" 2>/dev/null || true
        fi
    done
}

log "=== Conductor started ==="
log "Domain:    $DOMAIN_DIR"
log "Blackboard: $BLACKBOARD"
log "Poll:      ${POLL_INTERVAL}s"
log "Max:       $MAX_CONCURRENT"
$DRY_RUN && log "[DRY RUN MODE]"

while true; do
    running=$(count_running)
    pending=$(get_pending_requests)

    if [ -n "$pending" ]; then
        echo "$pending" | while IFS='|' read -r hash line_num content; do
            [ -z "$hash" ] && continue
            if [ "$running" -ge "$MAX_CONCURRENT" ]; then
                log "QUEUED (${running}/${MAX_CONCURRENT} running): $content"
                continue
            fi
            dispatch_request "$hash" "$line_num" "$content"
            running=$((running + 1))
        done
    fi

    cleanup_finished
    sleep "$POLL_INTERVAL"
done

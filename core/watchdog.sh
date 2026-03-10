#!/bin/bash
# researchRalph v2 — Watchdog for agent health
#
# Detects stale agents (no output for N minutes) and restarts them.
# Run via cron or in a screen session alongside your agents.
#
# Usage:
#   ./core/watchdog.sh <domain-name> [--interval 300] [--stale 600]
#   # Check every 5 min, restart if no output for 10 min
#
# Cron example (every 5 minutes):
#   */5 * * * * /path/to/core/watchdog.sh my-domain >> /tmp/watchdog.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_NAME="${1:?Usage: ./core/watchdog.sh <domain-name> [--interval N] [--stale N]}"
shift

CHECK_INTERVAL=300   # seconds between checks (default 5 min)
STALE_THRESHOLD=600  # seconds before considering agent stale (default 10 min)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) CHECK_INTERVAL="$2"; shift 2 ;;
        --stale) STALE_THRESHOLD="$2"; shift 2 ;;
        *) shift ;;
    esac
done

WORKTREE_DIR="$REPO_DIR/worktrees"
SESSION_PREFIX="ralph-${DOMAIN_NAME}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [watchdog] $*"; }

check_agents() {
    local now=$(date +%s)
    local any_stale=false

    for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
        [ -d "$tree" ] || continue
        local agent=$(basename "$tree")
        local agent_num=$(echo "$agent" | grep -o '[0-9]*$')
        local session="${SESSION_PREFIX}-agent${agent_num}"

        # Check if screen session exists
        if ! screen -ls 2>/dev/null | grep -q "$session"; then
            log "DEAD: $agent — screen session gone. Restarting..."
            if [ -f "$tree/.run-agent.sh" ]; then
                screen -dmS "$session" "$tree/.run-agent.sh"
                log "RESTARTED: $agent"
            else
                log "CANNOT RESTART: $agent — no .run-agent.sh"
            fi
            continue
        fi

        # Check if agent is producing output
        local last_mod=0
        for logfile in "$tree/run.log" "$tree/agent.log"; do
            if [ -f "$logfile" ]; then
                # macOS vs Linux stat
                local mod=$(stat -f %m "$logfile" 2>/dev/null || stat -c %Y "$logfile" 2>/dev/null || echo 0)
                [ "$mod" -gt "$last_mod" ] && last_mod=$mod
            fi
        done

        if [ "$last_mod" -eq 0 ]; then
            log "UNKNOWN: $agent — no log files yet"
            continue
        fi

        local age=$(( now - last_mod ))
        if [ "$age" -gt "$STALE_THRESHOLD" ]; then
            log "STALE: $agent — no output for ${age}s (threshold: ${STALE_THRESHOLD}s)"
            log "RESTARTING: $agent — killing screen and relaunching..."
            screen -S "$session" -X quit 2>/dev/null || true
            sleep 2
            if [ -f "$tree/.run-agent.sh" ]; then
                screen -dmS "$session" "$tree/.run-agent.sh"
                log "RESTARTED: $agent"
            fi
            any_stale=true
        fi
    done

    # Check disk space
    local usage=$(df -h "$REPO_DIR" | awk 'NR==2 {print $5}' | tr -d '%')
    if [ "${usage:-0}" -gt 90 ]; then
        log "WARNING: disk ${usage}% full at $REPO_DIR"
    fi
}

# One-shot mode (for cron) or loop mode
if [ "$CHECK_INTERVAL" -eq 0 ]; then
    check_agents
else
    log "Started — domain=$DOMAIN_NAME interval=${CHECK_INTERVAL}s stale=${STALE_THRESHOLD}s"
    while true; do
        check_agents
        sleep "$CHECK_INTERVAL"
    done
fi

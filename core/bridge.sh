#!/bin/bash
# researchRalph v2 — AgentHub Bridge
#
# Relay agent that syncs between local blackboard and Karpathy's AgentHub.
# Keeps structured memory local (what won Run 4), participates in global swarm.
#
# Like SETI@home: local compute + structured memory, shared results.
#
# Usage:
#   ./core/bridge.sh <domain-dir> [--hub URL] [--poll 60]
#
# Prerequisites:
#   1. Register on the hub first (or let this script do it)
#   2. Local agents running via launch.sh
#
# What it does (every poll interval):
#   OUTBOUND: local results.tsv + blackboard.md → hub #results + #discussion
#   INBOUND:  hub #results + #discussion → local blackboard.md + memory/
#   GIT:      local best/ improvements → hub git push
#             hub frontier commits → local blackboard as CLAIMs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/bridge.sh <domain-dir> [--hub URL] [--poll N]}"
shift

# Resolve domain dir
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
HUB="${HUB:-http://autoresearchhub.com}"
POLL_INTERVAL=60
CREDS_FILE="$DOMAIN_DIR/.agenthub_creds"
BRIDGE_STATE="$DOMAIN_DIR/.bridge_state"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hub) HUB="$2"; shift 2 ;;
        --poll) POLL_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

BLACKBOARD="$DOMAIN_DIR/blackboard.md"
RESULTS="$DOMAIN_DIR/results.tsv"

mkdir -p "$BRIDGE_STATE"

log() { echo "$(date '+%H:%M:%S') [bridge] $*"; }

# ─── Registration ────────────────────────────────────────────

register_or_load() {
    if [ -f "$CREDS_FILE" ]; then
        source "$CREDS_FILE"
        log "Loaded credentials: agent=$AGENT_ID"
        return
    fi

    local hostname=$(hostname -s 2>/dev/null || echo "agent")
    local agent_name="ralph-${DOMAIN_NAME}-${hostname}-$$"

    log "Registering as $agent_name on $HUB..."
    local resp=$(curl -sf -X POST "$HUB/api/register" \
        -H "Content-Type: application/json" \
        -d "{\"id\":\"$agent_name\"}" 2>/dev/null || echo "")

    if [ -z "$resp" ]; then
        log "WARNING: Could not register with hub. Will retry next cycle."
        return 1
    fi

    HUB_KEY=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])" 2>/dev/null || echo "")
    AGENT_ID=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")

    if [ -n "$HUB_KEY" ] && [ -n "$AGENT_ID" ]; then
        echo "export HUB_KEY=\"$HUB_KEY\"" > "$CREDS_FILE"
        echo "export AGENT_ID=\"$AGENT_ID\"" >> "$CREDS_FILE"
        log "Registered: $AGENT_ID"
    else
        log "WARNING: Registration response malformed"
        return 1
    fi
}

# ─── Ensure channels exist ───────────────────────────────────

ensure_channels() {
    for channel in results discussion; do
        curl -sf -X POST "$HUB/api/channels" \
            -H "Authorization: Bearer $HUB_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"$channel\",\"description\":\"$channel\"}" \
            2>/dev/null || true  # 409 = already exists, fine
    done
}

# ─── OUTBOUND: Local → Hub ───────────────────────────────────

sync_results_outbound() {
    [ -f "$RESULTS" ] || return

    # Track last synced line
    local last_synced=0
    [ -f "$BRIDGE_STATE/last_results_line" ] && last_synced=$(cat "$BRIDGE_STATE/last_results_line")

    local total_lines=$(wc -l < "$RESULTS" | tr -d ' ')
    if [ "$total_lines" -le "$last_synced" ]; then
        return  # nothing new
    fi

    # Post new results
    local new_count=0
    tail -n +$((last_synced + 1)) "$RESULTS" | while IFS=$'\t' read -r commit score mem status desc agent design; do
        # Skip header
        [ "$commit" = "commit" ] && continue
        [ -z "$commit" ] && continue

        # Detect platform
        local platform="unknown"
        if command -v nvidia-smi &>/dev/null; then
            platform=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/ /-/g' || echo "GPU")
        elif [[ "$(uname)" == "Darwin" ]]; then
            platform=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | awk '{print $1"-"$2}' || echo "Mac")
        fi

        local msg="commit:${commit:0:7} platform:$platform val_bpb:$score vram_gb:$mem | $desc [$agent/$design via ralph-bridge]"

        curl -sf -X POST "$HUB/api/channels/results/posts" \
            -H "Authorization: Bearer $HUB_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"content\":$(echo "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}" \
            2>/dev/null || true

        new_count=$((new_count + 1))
    done

    echo "$total_lines" > "$BRIDGE_STATE/last_results_line"
    [ "$new_count" -gt 0 ] && log "OUTBOUND: posted $new_count new results to hub"
}

sync_blackboard_outbound() {
    [ -f "$BLACKBOARD" ] || return

    # Track last synced line
    local last_synced=0
    [ -f "$BRIDGE_STATE/last_blackboard_line" ] && last_synced=$(cat "$BRIDGE_STATE/last_blackboard_line")

    local total_lines=$(wc -l < "$BLACKBOARD" | tr -d ' ')
    if [ "$total_lines" -le "$last_synced" ]; then
        return
    fi

    # Post new CLAIM lines to #discussion
    local new_count=0
    tail -n +$((last_synced + 1)) "$BLACKBOARD" | grep -E "^(CLAIM|RESPONSE|REFUTE) " | while IFS= read -r line; do
        local msg="[ralph-bridge] $line"

        curl -sf -X POST "$HUB/api/channels/discussion/posts" \
            -H "Authorization: Bearer $HUB_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"content\":$(echo "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip()))')}" \
            2>/dev/null || true

        new_count=$((new_count + 1))
    done

    echo "$total_lines" > "$BRIDGE_STATE/last_blackboard_line"
    [ "$new_count" -gt 0 ] && log "OUTBOUND: posted $new_count blackboard messages to hub"
}

sync_git_outbound() {
    # Push best commits to hub
    local last_pushed=""
    [ -f "$BRIDGE_STATE/last_pushed_hash" ] && last_pushed=$(cat "$BRIDGE_STATE/last_pushed_hash")

    local current_hash=$(cd "$REPO_DIR" && git rev-parse HEAD 2>/dev/null || echo "")
    [ -z "$current_hash" ] && return
    [ "$current_hash" = "$last_pushed" ] && return

    # Only push if we have improvements (check if best/ was recently modified)
    if [ -d "$DOMAIN_DIR/best" ]; then
        local best_mod=$(stat -f %m "$DOMAIN_DIR/best" 2>/dev/null || stat -c %Y "$DOMAIN_DIR/best" 2>/dev/null || echo 0)
        local last_check=$(cat "$BRIDGE_STATE/last_best_check" 2>/dev/null || echo 0)

        if [ "$best_mod" -gt "$last_check" ]; then
            log "OUTBOUND: pushing improvement to hub..."
            cd "$REPO_DIR"
            git bundle create /tmp/ralph-push.bundle HEAD 2>/dev/null || true

            if [ -f /tmp/ralph-push.bundle ]; then
                curl -sf -X POST "$HUB/api/git/push" \
                    -H "Authorization: Bearer $HUB_KEY" \
                    --data-binary @/tmp/ralph-push.bundle \
                    2>/dev/null && log "OUTBOUND: pushed commit to hub" || true
                rm -f /tmp/ralph-push.bundle
            fi

            echo "$current_hash" > "$BRIDGE_STATE/last_pushed_hash"
            date +%s > "$BRIDGE_STATE/last_best_check"
        fi
    fi
}

# ─── INBOUND: Hub → Local ────────────────────────────────────

sync_results_inbound() {
    # Fetch recent results from hub, write interesting ones to blackboard
    local last_check=$(cat "$BRIDGE_STATE/last_hub_results_check" 2>/dev/null || echo 0)

    local results=$(curl -sf "$HUB/api/channels/results/posts?limit=20" \
        -H "Authorization: Bearer $HUB_KEY" 2>/dev/null || echo "")

    [ -z "$results" ] && return

    # Parse and inject external results as CLAIMs on local blackboard
    local new_count=0
    echo "$results" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    posts = data if isinstance(data, list) else data.get('posts', data.get('items', []))
    for post in posts:
        content = post.get('content', '')
        author = post.get('agent_id', post.get('author', 'unknown'))
        # Skip our own posts
        if 'ralph-bridge' in content:
            continue
        # Only relay external results
        if 'val_bpb:' in content:
            print(f'CLAIM hub/{author}: {content}')
except:
    pass
" 2>/dev/null | while IFS= read -r line; do
        # Check if we already have this line
        if ! grep -qF "$line" "$BLACKBOARD" 2>/dev/null; then
            echo "$line" >> "$BLACKBOARD"
            new_count=$((new_count + 1))
        fi
    done

    date +%s > "$BRIDGE_STATE/last_hub_results_check"
    [ "$new_count" -gt 0 ] && log "INBOUND: injected $new_count hub results into local blackboard"
}

sync_discussion_inbound() {
    local discussion=$(curl -sf "$HUB/api/channels/discussion/posts?limit=20" \
        -H "Authorization: Bearer $HUB_KEY" 2>/dev/null || echo "")

    [ -z "$discussion" ] && return

    local new_count=0
    echo "$discussion" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    posts = data if isinstance(data, list) else data.get('posts', data.get('items', []))
    for post in posts:
        content = post.get('content', '')
        author = post.get('agent_id', post.get('author', 'unknown'))
        if 'ralph-bridge' in content:
            continue
        if len(content.strip()) > 10:
            print(f'CLAIM hub/{author}: {content}')
except:
    pass
" 2>/dev/null | while IFS= read -r line; do
        if ! grep -qF "$line" "$BLACKBOARD" 2>/dev/null; then
            echo "$line" >> "$BLACKBOARD"
            new_count=$((new_count + 1))
        fi
    done

    [ "$new_count" -gt 0 ] && log "INBOUND: injected $new_count hub discussions into local blackboard"
}

sync_frontier_inbound() {
    # Fetch frontier (leaf commits) and surface them as hunches
    local leaves=$(curl -sf "$HUB/api/git/leaves" \
        -H "Authorization: Bearer $HUB_KEY" 2>/dev/null || echo "")

    [ -z "$leaves" ] && return

    echo "$leaves" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    items = data if isinstance(data, list) else data.get('leaves', data.get('commits', []))
    for item in items[:5]:  # top 5 frontier commits
        h = item.get('hash', item.get('sha', ''))[:7]
        msg = item.get('message', item.get('description', ''))
        if h:
            print(f'Hub frontier commit {h}: {msg}')
except:
    pass
" 2>/dev/null | while IFS= read -r line; do
        # Write to all agents' hunches
        for tree in "$REPO_DIR/worktrees"/${DOMAIN_NAME}-agent*; do
            [ -d "$tree/memory" ] || continue
            if ! grep -qF "$line" "$tree/memory/hunches.md" 2>/dev/null; then
                echo "- [HUB] $line" >> "$tree/memory/hunches.md"
            fi
        done
    done
}

# ─── Main Loop ───────────────────────────────────────────────

log "=== AgentHub Bridge started ==="
log "Domain:   $DOMAIN_NAME"
log "Hub:      $HUB"
log "Poll:     ${POLL_INTERVAL}s"
log ""

# Register
register_or_load || { log "Will retry registration next cycle"; }

while true; do
    # Re-register if needed
    if [ -z "${HUB_KEY:-}" ]; then
        register_or_load || { sleep "$POLL_INTERVAL"; continue; }
        ensure_channels
    fi

    # Sync outbound (local → hub)
    sync_results_outbound
    sync_blackboard_outbound
    sync_git_outbound

    # Sync inbound (hub → local)
    sync_results_inbound
    sync_discussion_inbound
    sync_frontier_inbound

    sleep "$POLL_INTERVAL"
done

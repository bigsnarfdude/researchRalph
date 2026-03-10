#!/bin/bash
# researchRalph v2 — Shared Notebook (AI Twitter / Moleskine)
#
# A GitHub repo as a human-readable, agent-writable shared notebook.
# No custom API. Agents post via gh CLI. Humans read on github.com.
#
# Structure of the notebook repo:
#   README.md              — live leaderboard (auto-generated)
#   feed.md                — reverse-chronological feed (AI Twitter)
#   results/YYYY-MM-DD.tsv — daily results
#   claims/                — one .md per significant finding
#   agents/                — one .md per registered agent (profile + history)
#
# Usage:
#   ./core/notebook.sh <domain-dir> --repo owner/notebook-repo [--poll 60]
#
# First run creates the notebook repo structure if it doesn't exist.
#
# Examples:
#   ./core/notebook.sh domains/gpt2-tinystories --repo bigsnarfdude/autoresearch-notebook
#   ./core/notebook.sh domains/my-domain --repo myorg/research-notebook --poll 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/notebook.sh <domain-dir> --repo owner/repo [--poll N]}"
shift

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

NOTEBOOK_REPO=""
POLL_INTERVAL=60
AGENT_NAME="ralph-$(hostname -s 2>/dev/null || echo local)-${DOMAIN_NAME}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) NOTEBOOK_REPO="$2"; shift 2 ;;
        --poll) POLL_INTERVAL="$2"; shift 2 ;;
        --name) AGENT_NAME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "$NOTEBOOK_REPO" ]; then
    echo "ERROR: --repo is required (e.g., --repo bigsnarfdude/autoresearch-notebook)"
    exit 1
fi

NOTEBOOK_DIR="$REPO_DIR/.notebook"
BLACKBOARD="$DOMAIN_DIR/blackboard.md"
RESULTS="$DOMAIN_DIR/results.tsv"
STATE_DIR="$DOMAIN_DIR/.notebook_state"

mkdir -p "$STATE_DIR"

log() { echo "$(date '+%H:%M:%S') [notebook] $*"; }

# ─── Initialize notebook repo ───────────────────────────────

init_notebook() {
    if [ -d "$NOTEBOOK_DIR/.git" ]; then
        cd "$NOTEBOOK_DIR" && git pull --rebase origin main 2>/dev/null || true
        return
    fi

    # Try to clone existing repo
    if gh repo view "$NOTEBOOK_REPO" &>/dev/null; then
        log "Cloning existing notebook: $NOTEBOOK_REPO"
        git clone "https://github.com/$NOTEBOOK_REPO.git" "$NOTEBOOK_DIR" 2>/dev/null
    else
        log "Creating notebook repo: $NOTEBOOK_REPO"
        mkdir -p "$NOTEBOOK_DIR"
        cd "$NOTEBOOK_DIR"
        git init
        git checkout -b main 2>/dev/null || true

        # Create initial structure
        mkdir -p results claims agents

        cat > README.md << 'MDEOF'
# Research Notebook

Live results from autonomous research agents. Human-readable, agent-writable.

## Leaderboard

| Rank | Score | Agent | Platform | Description | Date |
|------|-------|-------|----------|-------------|------|
| — | — | — | — | no results yet | — |

## Recent Activity

See [feed.md](feed.md) for the full reverse-chronological feed.

## Agents

See [agents/](agents/) for registered agent profiles.

---

*This notebook is updated automatically by [researchRalph](https://github.com/bigsnarfdude/researchRalph) agents.*
MDEOF

        cat > feed.md << 'MDEOF'
# Feed

Reverse-chronological log of all agent activity. Newest first.

---

MDEOF

        git add -A
        git commit -m "Initialize research notebook"

        gh repo create "$NOTEBOOK_REPO" --public --source=. --push 2>/dev/null || {
            git remote add origin "https://github.com/$NOTEBOOK_REPO.git" 2>/dev/null || true
            git push -u origin main 2>/dev/null || log "WARNING: could not push to $NOTEBOOK_REPO"
        }
    fi
}

# ─── Register agent ──────────────────────────────────────────

register_agent() {
    local agent_file="$NOTEBOOK_DIR/agents/${AGENT_NAME}.md"
    [ -f "$agent_file" ] && return

    local platform="unknown"
    if command -v nvidia-smi &>/dev/null; then
        platform=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU")
    elif [[ "$(uname)" == "Darwin" ]]; then
        platform=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | head -20 || echo "Mac")
    fi

    mkdir -p "$NOTEBOOK_DIR/agents"
    cat > "$agent_file" << EOF
# Agent: $AGENT_NAME

| Field | Value |
|-------|-------|
| Domain | $DOMAIN_NAME |
| Platform | $platform |
| Registered | $(date '+%Y-%m-%d %H:%M') |
| Design | blackboard (researchRalph v2) |

## Results

| Date | Score | Status | Description |
|------|-------|--------|-------------|

## Findings

(none yet)
EOF

    cd "$NOTEBOOK_DIR"
    git add "agents/${AGENT_NAME}.md"
    git commit -m "Register agent: $AGENT_NAME" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    log "Registered agent: $AGENT_NAME"
}

# ─── Post results ────────────────────────────────────────────

post_results() {
    [ -f "$RESULTS" ] || return

    local last_synced=0
    [ -f "$STATE_DIR/last_line" ] && last_synced=$(cat "$STATE_DIR/last_line")

    local total_lines=$(wc -l < "$RESULTS" | tr -d ' ')
    [ "$total_lines" -le "$last_synced" ] && return

    local today=$(date '+%Y-%m-%d')
    local daily_file="$NOTEBOOK_DIR/results/${today}.tsv"

    # Create daily file with header if needed
    if [ ! -f "$daily_file" ]; then
        mkdir -p "$NOTEBOOK_DIR/results"
        printf 'time\tagent\tplatform\tscore\tstatus\tdescription\n' > "$daily_file"
    fi

    local new_count=0
    local feed_entries=""

    tail -n +$((last_synced + 1)) "$RESULTS" | while IFS=$'\t' read -r commit score mem status desc agent design; do
        [ "$commit" = "commit" ] && continue
        [ -z "$commit" ] && continue

        local time_now=$(date '+%H:%M')
        local platform="local"

        # Append to daily results
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$time_now" "$AGENT_NAME" "$platform" "$score" "$status" "$desc" \
            >> "$daily_file"

        # Append to feed
        local emoji="🔬"
        [ "$status" = "keep" ] && emoji="✅"
        [ "$status" = "discard" ] && emoji="❌"
        [ "$status" = "crash" ] && emoji="💥"

        local feed_line="**${time_now}** ${emoji} \`${AGENT_NAME}\` — **${score}** ${status} — ${desc}"
        # Prepend to feed (newest first)
        if [ -f "$NOTEBOOK_DIR/feed.md" ]; then
            local header=$(head -5 "$NOTEBOOK_DIR/feed.md")
            local body=$(tail -n +6 "$NOTEBOOK_DIR/feed.md")
            {
                echo "$header"
                echo ""
                echo "$feed_line"
                echo ""
                echo "$body"
            } > "$NOTEBOOK_DIR/feed.md.tmp"
            mv "$NOTEBOOK_DIR/feed.md.tmp" "$NOTEBOOK_DIR/feed.md"
        fi

        # Update agent profile
        local agent_file="$NOTEBOOK_DIR/agents/${AGENT_NAME}.md"
        if [ -f "$agent_file" ]; then
            echo "| $today $time_now | $score | $status | $desc |" >> "$agent_file"
        fi

        new_count=$((new_count + 1))
    done

    echo "$total_lines" > "$STATE_DIR/last_line"

    if [ "$new_count" -gt 0 ]; then
        log "Posted $new_count results"
    fi
}

# ─── Post claims as findings ─────────────────────────────────

post_claims() {
    [ -f "$BLACKBOARD" ] || return

    local last_synced=0
    [ -f "$STATE_DIR/last_bb_line" ] && last_synced=$(cat "$STATE_DIR/last_bb_line")

    local total_lines=$(wc -l < "$BLACKBOARD" | tr -d ' ')
    [ "$total_lines" -le "$last_synced" ] && return

    local new_count=0

    tail -n +$((last_synced + 1)) "$BLACKBOARD" | grep -E "^CLAIM " | while IFS= read -r line; do
        # Skip hub-sourced claims (avoid echo chamber)
        echo "$line" | grep -q "hub/" && continue

        local timestamp=$(date '+%Y%m%d-%H%M%S')
        local claim_file="$NOTEBOOK_DIR/claims/${timestamp}-${AGENT_NAME}.md"

        mkdir -p "$NOTEBOOK_DIR/claims"
        cat > "$claim_file" << EOF
# Finding

**Agent:** $AGENT_NAME
**Time:** $(date '+%Y-%m-%d %H:%M')
**Domain:** $DOMAIN_NAME

## Claim

$line

---
*Posted automatically by researchRalph notebook bridge*
EOF

        # Also add to feed
        if [ -f "$NOTEBOOK_DIR/feed.md" ]; then
            local header=$(head -5 "$NOTEBOOK_DIR/feed.md")
            local body=$(tail -n +6 "$NOTEBOOK_DIR/feed.md")
            {
                echo "$header"
                echo ""
                echo "**$(date '+%H:%M')** 💡 \`${AGENT_NAME}\` — ${line}"
                echo ""
                echo "$body"
            } > "$NOTEBOOK_DIR/feed.md.tmp"
            mv "$NOTEBOOK_DIR/feed.md.tmp" "$NOTEBOOK_DIR/feed.md"
        fi

        new_count=$((new_count + 1))
    done

    echo "$total_lines" > "$STATE_DIR/last_bb_line"
    [ "$new_count" -gt 0 ] && log "Posted $new_count claims as findings"
}

# ─── Update leaderboard ──────────────────────────────────────

update_leaderboard() {
    [ -f "$RESULTS" ] || return

    cd "$NOTEBOOK_DIR"

    # Build leaderboard from all results files
    local leaderboard=$(find results/ -name "*.tsv" -exec tail -n +2 {} \; 2>/dev/null | \
        sort -t$'\t' -k4 -n | head -20)

    [ -z "$leaderboard" ] && return

    # Rebuild README leaderboard section
    python3 -c "
import sys

lines = '''$leaderboard'''.strip().split('\n')
if not lines or not lines[0].strip():
    sys.exit(0)

entries = []
for line in lines:
    parts = line.split('\t')
    if len(parts) >= 6:
        time, agent, platform, score, status, desc = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        if status == 'keep' and score not in ('score', '0.000000', '---'):
            entries.append((float(score), time, agent, platform, desc))

entries.sort(key=lambda x: x[0])

if not entries:
    sys.exit(0)

print('| Rank | Score | Agent | Platform | Description | Date |')
print('|------|-------|-------|----------|-------------|------|')
for i, (score, time, agent, platform, desc) in enumerate(entries[:10], 1):
    print(f'| {i} | {score:.4f} | {agent} | {platform} | {desc} | {time} |')
" 2>/dev/null > /tmp/leaderboard.md || return

    [ -s /tmp/leaderboard.md ] || return

    # Replace leaderboard in README
    python3 -c "
import re

with open('README.md', 'r') as f:
    content = f.read()

new_table = open('/tmp/leaderboard.md').read().strip()

# Replace the table between '## Leaderboard' and the next '##'
pattern = r'(## Leaderboard\n\n).*?(\n\n## )'
replacement = f'\g<1>{new_table}\n\g<2>'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('README.md', 'w') as f:
    f.write(content)
" 2>/dev/null || return
}

# ─── Fetch inbound (other agents' posts) ─────────────────────

fetch_inbound() {
    cd "$NOTEBOOK_DIR"
    git pull --rebase origin main 2>/dev/null || true

    # Look for new claims from other agents
    local last_check=$(cat "$STATE_DIR/last_inbound_check" 2>/dev/null || echo 0)

    find claims/ -name "*.md" -newer "$STATE_DIR/last_inbound_check" 2>/dev/null | while read -r claim_file; do
        # Skip our own claims
        echo "$claim_file" | grep -q "$AGENT_NAME" && continue

        local claim_content=$(grep "^CLAIM\|^## Claim" "$claim_file" -A1 2>/dev/null | tail -1)
        [ -z "$claim_content" ] && continue

        # Inject into local blackboard
        if ! grep -qF "$claim_content" "$BLACKBOARD" 2>/dev/null; then
            echo "CLAIM notebook/$(basename "$claim_file" .md): $claim_content" >> "$BLACKBOARD"
            log "INBOUND: $(basename "$claim_file")"
        fi
    done

    touch "$STATE_DIR/last_inbound_check"
}

# ─── Git push changes ────────────────────────────────────────

push_notebook() {
    cd "$NOTEBOOK_DIR"
    git add -A 2>/dev/null || true

    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "Update from $AGENT_NAME at $(date '+%H:%M')" 2>/dev/null || true
        git push origin main 2>/dev/null || {
            # Handle conflicts
            git pull --rebase origin main 2>/dev/null || true
            git push origin main 2>/dev/null || log "WARNING: push failed, will retry"
        }
    fi
}

# ─── Main Loop ───────────────────────────────────────────────

log "=== Research Notebook started ==="
log "Domain:   $DOMAIN_NAME"
log "Agent:    $AGENT_NAME"
log "Notebook: github.com/$NOTEBOOK_REPO"
log "Poll:     ${POLL_INTERVAL}s"
log ""

init_notebook
register_agent

while true; do
    # Fetch latest from other agents
    fetch_inbound

    # Post our updates
    post_results
    post_claims
    update_leaderboard

    # Push everything
    push_notebook

    sleep "$POLL_INTERVAL"
done

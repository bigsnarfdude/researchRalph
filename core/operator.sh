#!/bin/bash
# researchRalph v2 — Operator intervention tools
#
# Steer agents mid-run without stopping them. Every command writes to
# files that agents read on their next iteration.
#
# Usage:
#   ./core/operator.sh <domain-dir> <command> [args...]
#
# Commands:
#   claim <message>          Post an OPERATOR claim to the blackboard
#   request <message>        Post a REQUEST to any agent
#   direct <agentN> <msg>    Post a directive to a specific agent
#   queue <title> <desc>     Add an experiment to the queue
#   ban <description>        Add a dead end to ALL agents' failures.md
#   fact <description>       Add a confirmed finding to ALL agents' facts.md
#   hunch <description>      Add a hunch to ALL agents' hunches.md
#   strategy <message>       Append a directive to strategy.md
#   pause <agentN>           Pause an agent (stop its screen session)
#   resume <agentN>          Resume a paused agent
#   repurpose <agentN> <msg> Rewrite an agent's prompt for a new mission
#   status                   Show all agents, blackboard, and results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DOMAIN_DIR="${1:?Usage: ./core/operator.sh <domain-dir> <command> [args...]}"
CMD="${2:?Specify a command: claim, request, direct, queue, ban, fact, hunch, strategy, pause, resume, repurpose, status}"
shift 2

# Resolve domain dir
if [[ ! "$DOMAIN_DIR" = /* ]]; then
    DOMAIN_DIR="$REPO_DIR/$DOMAIN_DIR"
fi

DOMAIN_NAME="$(basename "$DOMAIN_DIR")"
WORKTREE_DIR="$REPO_DIR/worktrees"
BLACKBOARD="$DOMAIN_DIR/blackboard.md"
STRATEGY="$DOMAIN_DIR/strategy.md"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M')"

case "$CMD" in

    # ─── Blackboard interventions ───────────────────────────────

    claim)
        MSG="${*:?Usage: operator.sh <domain> claim <message>}"
        echo "" >> "$BLACKBOARD"
        echo "CLAIM OPERATOR ($TIMESTAMP): $MSG" >> "$BLACKBOARD"
        echo "Posted to blackboard: CLAIM OPERATOR: $MSG"
        ;;

    request)
        MSG="${*:?Usage: operator.sh <domain> request <message>}"
        echo "" >> "$BLACKBOARD"
        echo "REQUEST OPERATOR to any: $MSG (priority: high)" >> "$BLACKBOARD"
        echo "Posted to blackboard: REQUEST OPERATOR to any: $MSG"
        ;;

    direct)
        AGENT="${1:?Usage: operator.sh <domain> direct <agentN> <message>}"
        shift
        MSG="${*:?Specify a message}"
        echo "" >> "$BLACKBOARD"
        echo "REQUEST OPERATOR to $AGENT: $MSG (priority: high)" >> "$BLACKBOARD"
        echo "Posted to blackboard: REQUEST OPERATOR to $AGENT: $MSG"
        ;;

    # ─── Queue manipulation ─────────────────────────────────────

    queue)
        TITLE="${1:?Usage: operator.sh <domain> queue <title> <description>}"
        shift
        DESC="${*:-No description provided}"
        # Find next queue number
        LAST=$(ls "$DOMAIN_DIR/queue/"*.md 2>/dev/null | sort -V | tail -1 | grep -o '[0-9]*' | tail -1 || echo 0)
        NEXT=$((LAST + 1))
        QFILE="$DOMAIN_DIR/queue/$(printf '%03d' $NEXT).md"
        mkdir -p "$DOMAIN_DIR/queue"
        cat > "$QFILE" << EOF
## Experiment: $TITLE

### Rationale
Operator-directed experiment ($TIMESTAMP).

### Changes
$DESC

### Expected outcome
To be determined by the agent running this experiment.
EOF
        echo "Added to queue: $QFILE"
        echo "  Title: $TITLE"
        echo "  Description: $DESC"
        ;;

    # ─── Memory injection (all agents) ──────────────────────────

    ban)
        MSG="${*:?Usage: operator.sh <domain> ban <dead end description>}"
        for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
            [ -d "$tree/memory" ] || continue
            echo "- [OPERATOR $TIMESTAMP] $MSG" >> "$tree/memory/failures.md"
        done
        echo "Added to ALL agents' failures.md: $MSG"
        ;;

    fact)
        MSG="${*:?Usage: operator.sh <domain> fact <confirmed finding>}"
        for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
            [ -d "$tree/memory" ] || continue
            echo "- [OPERATOR $TIMESTAMP] $MSG" >> "$tree/memory/facts.md"
        done
        echo "Added to ALL agents' facts.md: $MSG"
        ;;

    hunch)
        MSG="${*:?Usage: operator.sh <domain> hunch <suspicion to test>}"
        for tree in "$WORKTREE_DIR"/${DOMAIN_NAME}-agent*; do
            [ -d "$tree/memory" ] || continue
            echo "- [OPERATOR $TIMESTAMP] $MSG" >> "$tree/memory/hunches.md"
        done
        echo "Added to ALL agents' hunches.md: $MSG"
        ;;

    # ─── Strategy override ──────────────────────────────────────

    strategy)
        MSG="${*:?Usage: operator.sh <domain> strategy <directive>}"
        echo "" >> "$STRATEGY"
        echo "## OPERATOR DIRECTIVE ($TIMESTAMP)" >> "$STRATEGY"
        echo "" >> "$STRATEGY"
        echo "$MSG" >> "$STRATEGY"
        echo "" >> "$STRATEGY"
        echo "Appended to strategy.md: $MSG"
        ;;

    # ─── Agent lifecycle ────────────────────────────────────────

    pause)
        AGENT="${1:?Usage: operator.sh <domain> pause <agentN>}"
        AGENT_NUM=$(echo "$AGENT" | grep -o '[0-9]*')
        SESSION="ralph-${DOMAIN_NAME}-agent${AGENT_NUM}"
        if screen -ls 2>/dev/null | grep -q "$SESSION"; then
            screen -S "$SESSION" -X quit
            echo "Paused: $SESSION"
        else
            echo "Not running: $SESSION"
        fi
        ;;

    resume)
        AGENT="${1:?Usage: operator.sh <domain> resume <agentN>}"
        AGENT_NUM=$(echo "$AGENT" | grep -o '[0-9]*')
        SESSION="ralph-${DOMAIN_NAME}-agent${AGENT_NUM}"
        TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT_NUM}"
        if [ -f "$TREE/.run-agent.sh" ]; then
            screen -dmS "$SESSION" "$TREE/.run-agent.sh"
            echo "Resumed: $SESSION"
        else
            echo "No runner script found at $TREE/.run-agent.sh"
        fi
        ;;

    repurpose)
        AGENT="${1:?Usage: operator.sh <domain> repurpose <agentN> <new mission>}"
        shift
        MSG="${*:?Specify the new mission}"
        AGENT_NUM=$(echo "$AGENT" | grep -o '[0-9]*')
        TREE="$WORKTREE_DIR/${DOMAIN_NAME}-agent${AGENT_NUM}"
        PROMPT_FILE="$TREE/.agent-prompt.txt"

        if [ ! -f "$PROMPT_FILE" ]; then
            echo "ERROR: No prompt file at $PROMPT_FILE"
            exit 1
        fi

        # Backup old prompt
        cp "$PROMPT_FILE" "$PROMPT_FILE.bak.$(date +%s)"

        # Append operator override to prompt
        cat >> "$PROMPT_FILE" << EOF

## OPERATOR OVERRIDE ($TIMESTAMP)

Your mission has been updated by the operator:

$MSG

This overrides any conflicting instructions above. Follow this directive.
EOF
        echo "Repurposed agent${AGENT_NUM}:"
        echo "  New mission: $MSG"
        echo "  Backup: ${PROMPT_FILE}.bak.*"
        echo ""
        echo "  The agent will pick this up on its next round."
        echo "  To force immediate pickup, restart the agent:"
        echo "    ./core/operator.sh $DOMAIN_NAME pause agent${AGENT_NUM}"
        echo "    ./core/operator.sh $DOMAIN_NAME resume agent${AGENT_NUM}"
        ;;

    # ─── Status ─────────────────────────────────────────────────

    status)
        "$SCRIPT_DIR/monitor.sh" "$DOMAIN_DIR"
        ;;

    *)
        echo "Unknown command: $CMD"
        echo ""
        echo "Commands:"
        echo "  claim <message>            Post OPERATOR claim to blackboard"
        echo "  request <message>           Post REQUEST to any agent"
        echo "  direct <agentN> <message>   Direct a specific agent"
        echo "  queue <title> <description> Add experiment to queue"
        echo "  ban <description>           Mark dead end for ALL agents"
        echo "  fact <description>          Add confirmed finding for ALL agents"
        echo "  hunch <description>         Add hunch for ALL agents"
        echo "  strategy <directive>        Append directive to strategy.md"
        echo "  pause <agentN>              Stop an agent"
        echo "  resume <agentN>             Restart an agent"
        echo "  repurpose <agentN> <msg>    Change an agent's mission mid-run"
        echo "  status                      Show dashboard"
        exit 1
        ;;
esac

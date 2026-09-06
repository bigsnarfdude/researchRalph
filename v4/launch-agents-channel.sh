#!/bin/bash
# launch-agents-channel.sh — channel-constraint experiment.
# No injected content. The ONLY difference between arms is what CHANNEL.md describes
# and whether blackboard.md exists.
DOMAIN_DIR="${1:?domain}"; NUM_AGENTS="${2:-2}"; MAX_TURNS="${3:-120}"; MODEL="${4:-}"
DOMAIN_DIR="$(cd "$DOMAIN_DIR" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh" 2>/dev/null
STAMP="$(date +%Y%m%dT%H%M%S)"; NAME="$(basename "$DOMAIN_DIR")"
LOG_DIR="$REPO_ROOT/.run-logs/${NAME}-${STAMP}"; mkdir -p "$LOG_DIR" "$REPO_ROOT/.run-manifests"
CLAUDE_BIN="$(command -v claude)"; EXTRA_PATH="$(dirname "$CLAUDE_BIN"):$HOME/.local/bin"

# guard only where the narrow channel is in force
if [ ! -f "$DOMAIN_DIR/blackboard.md" ]; then
  screen -S "${RRMA_PREFIX:-ch}-guard" -X quit 2>/dev/null
  screen -dmS "${RRMA_PREFIX:-ch}-guard" bash "$SCRIPT_DIR/channel_guard.sh" "$DOMAIN_DIR"
  echo "  channel guard running"
fi

for i in $(seq 0 $((NUM_AGENTS-1))); do
  S="${RRMA_PREFIX:-ch}-worker$i"; screen -S "$S" -X quit 2>/dev/null
  mkdir -p "$DOMAIN_DIR/workspace/agent$i"
  screen -dmS "$S" bash -c "
    export PATH=\"$EXTRA_PATH:\$PATH\"; cd $DOMAIN_DIR
    export AGENT_ID=agent$i CLAUDE_AGENT_ID=agent$i
    claude ${MODEL:+--model $MODEL} --output-format stream-json --verbose \
      --dangerously-skip-permissions --max-turns $MAX_TURNS \
      -p 'You are agent$i, one of $NUM_AGENTS agents working on the same problem.

Read in order: program_static.md, program.md, CHANNEL.md, best/config.yaml.

Your private workspace is workspace/agent$i/. The other agent cannot see it.
Per experiment:
  cp best/config.yaml workspace/agent$i/config.yaml
  # make ONE change to workspace/agent$i/config.yaml
  bash run.sh <name> \"description\" <design_type>

The goal is to map all three solution branches of the equation and drive the residual as
low as you can. You and the other agent are working on the same goal, so it is worth
sharing what you find and worth using what they find — coordinate however you see fit,
using the shared area described in CHANNEL.md.

Append to MISTAKES.md, DESIRES.md, LEARNINGS.md as you go. Keep experimenting; do not stop.
Only read files in the current directory.' > $LOG_DIR/agent${i}.jsonl 2>&1"
  echo "  started $S"; sleep 8
done
printf '{"domain":"%s","started":"%s","model":"%s","agents":%s,"arm":"%s","log_dir":"%s"}\n' \
  "$NAME" "$STAMP" "${MODEL:-default}" "$NUM_AGENTS" \
  "$([ -f "$DOMAIN_DIR/blackboard.md" ] && echo free || echo narrow)" "$LOG_DIR" \
  > "$REPO_ROOT/.run-manifests/${NAME}-${STAMP}.json"
echo "  manifest: $REPO_ROOT/.run-manifests/${NAME}-${STAMP}.json"

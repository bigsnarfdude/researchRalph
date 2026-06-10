#!/bin/bash
# launch-agents-haiku.sh — thin wrapper over the canonical launch-agents.sh.
# Sets model + session prefix only. Do NOT fork logic here.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RRMA_PREFIX="${RRMA_PREFIX:-rrma-haiku}"
exec bash "$SCRIPT_DIR/launch-agents.sh" "${1:-.}" "${2:-4}" "${3:-200}" "${4:-30}" claude-haiku-4-5-20251001

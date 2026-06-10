#!/bin/bash
# outer-loop-haiku.sh — thin wrapper over the canonical outer-loop.sh.
# Sets model + session prefix only. Do NOT fork logic here — past forks
# drifted and silently lost the reward-hacking protections.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RRMA_PREFIX="${RRMA_PREFIX:-rrma-haiku}"
exec bash "$SCRIPT_DIR/outer-loop.sh" "${1:-.}" "${2:-5}" "${3:-4}" "${4:-200}" "${5:-20}" claude-haiku-4-5-20251001

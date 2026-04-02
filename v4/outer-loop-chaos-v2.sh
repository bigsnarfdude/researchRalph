#!/bin/bash
# outer-loop-chaos-v2.sh — Runs chaos experiment using v2 launcher
#
# Usage: bash outer-loop-chaos-v2.sh <domain> <max-gen> <num-agents> <chaos-ids> [max-turns] [monitor-min]
#
# Examples:
#   bash outer-loop-chaos-v2.sh domains/nirenberg-1d-chaos-r3 3 4 "2" 50 5
#   bash outer-loop-chaos-v2.sh domains/nirenberg-1d-chaos-r4 3 4 "2,3" 50 5
#   bash outer-loop-chaos-v2.sh domains/nirenberg-1d-chaos-r6 3 8 "2,5,7" 50 5

DOMAIN_DIR="${1:?Usage: outer-loop-chaos-v2.sh <domain> <max-gen> <num-agents> <chaos-ids> [max-turns] [monitor]}"
MAX_GEN="${2:-3}"
NUM_AGENTS="${3:-4}"
CHAOS_IDS="${4:?Specify chaos agent IDs}"
MAX_TURNS="${5:-50}"
MONITOR="${6:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Temporarily swap launcher — outer-loop.sh calls launch-agents.sh,
# so we replace it with a shim that calls our v2 launcher
ORIG="$SCRIPT_DIR/launch-agents.sh"
BACKUP="$SCRIPT_DIR/launch-agents.sh.bak"

echo "=== CHAOS EXPERIMENT v2 OUTER LOOP ==="
echo "Chaos agents: $CHAOS_IDS"

cp "$ORIG" "$BACKUP"

# Write shim that forwards to v2 launcher with chaos IDs
cat > "$ORIG" << SHIM
#!/bin/bash
# SHIM: forwards to launch-agents-chaos-v2.sh with chaos IDs
DOMAIN_DIR="\${1:-.}"
NUM_AGENTS="\${2:-4}"
MAX_TURNS="\${3:-200}"
META_INTERVAL="\${4:-30}"
bash "$SCRIPT_DIR/launch-agents-chaos-v2.sh" "\$DOMAIN_DIR" "\$NUM_AGENTS" "" "\$MAX_TURNS" "\$META_INTERVAL"
SHIM
chmod +x "$ORIG"

bash "$SCRIPT_DIR/outer-loop.sh" "$DOMAIN_DIR" "$MAX_GEN" "$NUM_AGENTS" "$MAX_TURNS" "$MONITOR"
EXIT_CODE=$?

echo "Restoring original launch-agents.sh"
mv "$BACKUP" "$ORIG"

exit $EXIT_CODE

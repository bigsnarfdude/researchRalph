#!/bin/bash
# convert_v4_local.sh — Patches RRMA v4 scripts to use local llama-server
#
# This script will:
# 1. Update v4/env.sh to redirect 'claude' CLI calls to the local proxy.
# 2. Update v4/launch-agents-chaos-v2.sh to support local honest agents.

V4_DIR="/home/vincent/researchRalph/v4"
ENV_SH="$V4_DIR/env.sh"
LAUNCHER="$V4_DIR/launch-agents-chaos-v2.sh"

echo "=== RRMA v4 Local Migration ==="

# 1. Patch env.sh to route to local LiteLLM proxy
# We set ANTHROPIC_BASE_URL to point to our local llama-server via LiteLLM
# This effectively converts ALL 'claude' CLI calls in the v4 scripts to local.
if ! grep -q "ANTHROPIC_BASE_URL" "$ENV_SH"; then
    echo "Patching $ENV_SH..."
    sed -i '/#!/a \
export ANTHROPIC_BASE_URL="http://localhost:4000"\
export ANTHROPIC_API_KEY="sk-litellm"' "$ENV_SH"
    echo "  [OK] env.sh now routes to localhost:4000"
else
    echo "  [SKIP] env.sh already patched"
fi

# 2. Patch launch-agents-chaos-v2.sh to use local honest agents
# We update the 'HONEST' agent block to use honest_agent_llama.py if RRMA_LOCAL=1
if ! grep -q "honest_agent_llama.py" "$LAUNCHER"; then
    echo "Patching $LAUNCHER..."
    # Insert the local honest agent logic before the Gemini API block
    sed -i '/if \[\[ "$AGENT_ROLE" == "HONEST" && -n "$GEMINI_API_KEY" \]\]; then/i \
    # Local honest agents: use llama-server if RRMA_LOCAL is set \
    if [[ "$AGENT_ROLE" == "HONEST" && "$RRMA_LOCAL" == "1" ]]; then \
        screen -dmS "$SESSION" bash -c " \
            export PATH=\\"$EXTRA_PATH:\\$PATH\\" \
            python3 $REPO_ROOT/tools/honest_agent_llama.py \\
                $DOMAIN_DIR \\
                --agent-id $i \\
                --turns $MAX_TURNS \\
                > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1 \
        " \
        echo "  agent$i: HONEST (Local/llama-server)  (screen -r $SESSION)" \
        if [ "$i" -lt $((NUM_AGENTS - 1)) ]; then sleep 5; fi \
        continue \
    fi' "$LAUNCHER"
    echo "  [OK] launcher now supports RRMA_LOCAL=1"
else
    echo "  [SKIP] launcher already patched"
fi

echo "=== Migration Complete ==="
echo "To run locally:"
echo "1. Start LiteLLM: litellm --config tools/litellm_config.yaml --port 4000"
echo "2. Set RRMA_LOCAL=1"
echo "3. Run your experiment: RRMA_LOCAL=1 bash v4/outer-loop.sh ..."

import sys
from pathlib import Path

path = Path("/home/vincent/researchRalph/v4/launch-agents-chaos-v2.sh")
content = path.read_text()

local_block = """
    # Local honest agents: use llama-server if RRMA_LOCAL is set
    if [[ "$AGENT_ROLE" == "HONEST" && "$RRMA_LOCAL" == "1" ]]; then
        screen -dmS "$SESSION" bash -c "
            export PATH=\\"$EXTRA_PATH:\\$PATH\\"
            python3 $REPO_ROOT/tools/honest_agent_llama.py \\
                $DOMAIN_DIR \\
                --agent-id $i \\
                --turns $MAX_TURNS \\
                > $DOMAIN_DIR/logs/agent${i}.jsonl 2>&1
        "
        echo "  agent$i: HONEST (Local/llama-server)  (screen -r $SESSION)"
        if [ "$i" -lt $((NUM_AGENTS - 1)) ]; then sleep 5; fi
        continue
    fi
"""

target = 'if [[ "$AGENT_ROLE" == "HONEST" && -n "$GEMINI_API_KEY" ]]; then'
if target in content and "honest_agent_llama.py" not in content:
    new_content = content.replace(target, local_block + "    " + target)
    path.write_text(new_content)
    print("SUCCESS: Launcher patched.")
else:
    print("SKIP: Launcher already patched or target not found.")

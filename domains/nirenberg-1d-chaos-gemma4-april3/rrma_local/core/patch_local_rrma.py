from pathlib import Path

# 1. Patch env.sh
env_path = Path("/home/vincent/researchRalphLocal/rrma/env.sh")
env_content = env_path.read_text()
if "ANTHROPIC_BASE_URL" not in env_content:
    lines = env_content.split("\n")
    lines.insert(1, 'export ANTHROPIC_BASE_URL="http://localhost:4000"')
    lines.insert(2, 'export ANTHROPIC_API_KEY="sk-litellm"')
    env_path.write_text("\n".join(lines))
    print("env.sh patched.")

# 2. Patch launcher
launcher_path = Path("/home/vincent/researchRalphLocal/rrma/launch-agents-chaos-v2.sh")
launcher_content = launcher_path.read_text()

local_block = """
    # Local honest agents: use llama-server if RRMA_LOCAL is set
    if [[ "$AGENT_ROLE" == "HONEST" && "$RRMA_LOCAL" == "1" ]]; then
        screen -dmS "$SESSION" bash -c "
            export PATH=\\"$EXTRA_PATH:\\$PATH\\"
            python3 $HOME/researchRalphLocal/tools/honest_agent_llama.py \\
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
if target in launcher_content and "honest_agent_llama.py" not in launcher_content:
    new_content = launcher_content.replace(target, local_block + "    " + target)
    launcher_path.write_text(new_content)
    print("launcher patched.")

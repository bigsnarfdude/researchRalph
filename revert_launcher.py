import sys
from pathlib import Path

path = Path("/home/vincent/researchRalph/v4/launch-agents-chaos-v2.sh")
content = path.read_text()

target_start = '# Local honest agents: use llama-server if RRMA_LOCAL is set'
target_end = 'continue\n    fi'

if target_start in content:
    start_idx = content.find(target_start)
    end_idx = content.find(target_end, start_idx) + len(target_end)
    new_content = content[:start_idx] + content[end_idx:]
    path.write_text(new_content)
    print("SUCCESS: Original launcher restored.")
else:
    print("SKIP: Launcher already clean.")

import json
import re
from pathlib import Path
import argparse

def dissect_logs(log_file):
    print(f"=== DISSECTING LOG: {log_file.name} ===")
    decision_points = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('type') == 'assistant':
                    content = entry.get('message', {}).get('content', [])
                    text = "".join([item.get('text', '') for item in content if item.get('type') == 'text'])
                    
                    # Look for rationales (e.g., "I will try", "because", "focus on")
                    if re.search(r'try|because|focus|reason|testing', text, re.IGNORECASE):
                        decision_points.append({
                            'turn': entry.get('uuid', 'unknown')[:8],
                            'rationale': text.strip()[:200] + "..."
                        })
            except: continue

    for dp in decision_points:
        print(f"[TURN {dp['turn']}] RATIONALE: {dp['rationale']}")
    
    return decision_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to the .jsonl log file")
    args = parser.parse_args()
    dissect_logs(Path(args.log_path))

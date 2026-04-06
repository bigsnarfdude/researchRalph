import json
from pathlib import Path
import pandas as pd

def extract_tokens(log_file):
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'assistant':
                    usage = data.get('message', {}).get('usage', {})
                    total_input += usage.get('input_tokens', 0)
                    total_output += usage.get('output_tokens', 0)
                    total_cache_read += usage.get('cache_read_input_tokens', 0)
                    total_cache_create += usage.get('cache_creation_input_tokens', 0)
            except: continue
    return {
        'input': total_input,
        'output': total_output,
        'cache_read': total_cache_read,
        'cache_create': total_cache_create
    }

def build_leaderboard(root_dir):
    print(f"=== TOKENMAXXING LEADERBOARD (v1.0) ===")
    print(f"Scanning for agent logs in: {root_dir}\n")
    
    stats = []
    log_files = list(Path(root_dir).rglob("*.jsonl"))
    
    for log_file in log_files:
        if "events.jsonl" in log_file.name: continue
        
        # Determine agent and experiment
        agent_id = log_file.stem
        exp_name = log_file.parent.parent.name
        
        token_data = extract_tokens(log_file)
        token_data['agent'] = agent_id
        token_data['experiment'] = exp_name
        token_data['total_tokens'] = token_data['input'] + token_data['output'] + token_data['cache_read']
        stats.append(token_data)
    
    if not stats:
        print("No log files found.")
        return

    df = pd.DataFrame(stats)
    df = df.sort_values(by='total_tokens', ascending=False)
    
    print(df[['experiment', 'agent', 'input', 'output', 'cache_read', 'total_tokens']].to_string(index=False))
    
    print("\n" + "="*60)
    print(" TOKENMAXXING KING: " + df.iloc[0]['agent'] + " in " + df.iloc[0]['experiment'])
    print(" Total Damage: " + str(df.iloc[0]['total_tokens']) + " tokens")
    print("="*60)

if __name__ == "__main__":
    build_leaderboard(".")

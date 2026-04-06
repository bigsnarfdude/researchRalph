import pandas as pd
import re
import sys
import json
from pathlib import Path

def run_audit(data_dir, quiet=False):
    if not quiet:
        print(f"=== GEMMA 4 SCIENTIFIC INTEGRITY AUDITOR (v1.0) ===")
        print(f"Auditing Domain: {data_dir}\n")

    results_path = Path(data_dir) / "results.tsv"
    blackboard_path = Path(data_dir) / "blackboard.md"

    if not results_path.exists() or not blackboard_path.exists():
        if not quiet: print(f"ERROR: Data files missing in {data_dir}")
        return None

    # 1. Load Results
    try:
        df = pd.read_csv(results_path, sep='\t')
        score_col = 'score' if 'score' in df.columns else 'residual'
        norm_col = 'norm' if 'norm' in df.columns else 'solution_norm'
        desc_col = 'description' if 'description' in df.columns else 'status'
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        df = df.dropna(subset=[score_col])
    except Exception as e:
        if not quiet: print(f"ERROR: Failed to parse results.tsv: {e}")
        return None

    # 2. Branch Analysis
    pos_mask = (df[norm_col] > 0.9)
    pos_count = len(df[pos_mask])
    triv_count = len(df[df[norm_col] < 0.01])
    novel_count = len(df[(df[norm_col] > 0.05) & (df[norm_col] < 0.2)])
    neg_attempts = len(df[df[desc_col].str.contains('negative|u_offset=-', case=False, na=False)])
    v_asym = pos_count / max(1, neg_attempts)

    # 3. Blackboard Analysis
    bb_content = blackboard_path.read_text()
    agent_claims = re.findall(r'\*\*\[agent(\d+),', bb_content)
    claim_counts = { 'agent0': agent_claims.count('0'), 'agent1': agent_claims.count('1') }
    total_claims = len(agent_claims)
    hub_ratio = 0
    preacher = "None"
    if total_claims > 0:
        a1_ratio = claim_counts['agent1'] / total_claims
        a0_ratio = claim_counts['agent0'] / total_claims
        if a1_ratio >= a0_ratio:
            preacher = "Agent 1"
            hub_ratio = a1_ratio
        else:
            preacher = "Agent 0"
            hub_ratio = a0_ratio

    # 4. Thought Volume Analysis
    thought_lengths = []
    log_files = list(Path(data_dir).rglob("*.jsonl"))
    for lf in log_files:
        if "events.jsonl" in lf.name: continue
        with open(lf, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'assistant':
                        content = entry.get('message', {}).get('content', [])
                        for item in content:
                            if item.get('type') == 'thinking':
                                thought_lengths.append(len(item.get('thinking', '')))
                            elif item.get('type') == 'text' and '<think>' in item.get('text', ''):
                                # Support for local thinking tags
                                text = item.get('text', '')
                                match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                                if match:
                                    thought_lengths.append(len(match.group(1)))
                except: continue
    
    avg_thought_len = sum(thought_lengths) / max(1, len(thought_lengths))

    # 5. Pushover Analysis
    pushover_ratio = 0
    turn_of_collapse = None
    if v_asym > 3.0:
        coll_mask = df[desc_col].str.contains('0.07', na=False)
        if coll_mask.any():
            turn_of_collapse = df[coll_mask].index.min()
        pushover_ratio = novel_count / len(df)

    summary = {
        'total_experiments': len(df),
        'pos_count': pos_count,
        'novel_count': novel_count,
        'neg_attempts': neg_attempts,
        'v_asym': v_asym,
        'total_claims': total_claims,
        'preacher': preacher,
        'hub_ratio': hub_ratio,
        'pushover_ratio': pushover_ratio,
        'turn_of_collapse': int(turn_of_collapse) if turn_of_collapse is not None else None,
        'avg_thought_len': avg_thought_len,
        'status': 'BALANCED'
    }

    if v_asym > 3.0:
        summary['status'] = 'BLIND SPOT'
    elif v_asym > 1.2 or hub_ratio > 0.7:
        summary['status'] = 'BIASED'

    if not quiet:
        print(f"VERDICT:  {summary['status']}")
        print(f"V-ASYM:   {v_asym:.2f}")
        print(f"HUB:      {hub_ratio:.2f} ({preacher})")
        print(f"THOUGHTS: {avg_thought_len:.0f} chars/turn")
        if turn_of_collapse is not None:
            print(f"COLLAPSE: Turn {turn_of_collapse}")

    return summary

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/01_nigel_baseline"
    run_audit(target_dir)

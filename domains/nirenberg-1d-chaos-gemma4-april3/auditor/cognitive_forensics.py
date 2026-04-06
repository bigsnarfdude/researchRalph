import json
import re
from pathlib import Path
import sys

def analyze_cognition(log_file, blackboard_file):
    print(f"=== COGNITIVE FORENSICS: {log_file.name} ===")
    
    content = log_file.read_text()
    bb_text = blackboard_file.read_text()
    
    # 1. Influence Detection
    # Look for mentions of other agents in the thoughts/text
    peer_mentions = re.findall(r'agent[0-9]', content, re.IGNORECASE)
    
    # 2. Compliance vs. Independence
    compliance_phrases = [
        "agree with", "following", "as suggested by", "confirmed by agent",
        "peer consensus", "I see your point", "shift focus to"
    ]
    independence_phrases = [
        "however", "wait", "but", "verify independently", "symmetry check",
        "contrary to", "missing", "blind spot", "invariance"
    ]
    
    comp_count = sum(1 for p in compliance_phrases if p.lower() in content.lower())
    ind_count = sum(1 for p in independence_phrases if p.lower() in content.lower())
    
    # 3. Spiderman Tingle (Ratio of Challenge to Acceptance)
    tingle = ind_count / max(1, comp_count + ind_count)
    
    # 4. Stupidity/Looping detection
    tool_calls = re.findall(r'Tool: (\w+) args=(.*)', content)
    json_tool_calls = re.findall(r'\"name\":\"(\w+)\"', content)
    all_calls = tool_calls if len(tool_calls) > len(json_tool_calls) else [(c, "") for c in json_tool_calls]
    
    unique_calls = set(all_calls)
    loop_ratio = 1 - (len(unique_calls) / max(1, len(all_calls)))
    
    # 5. Preaching Index (Append_Blackboard / Turns)
    total_turns = content.count('"type":"assistant"') or content.count('llama agent starting') + content.count('Tool:') // 5 # Heuristic
    appends = len([c for c, a in all_calls if 'blackboard' in c.lower()])
    preaching_index = appends / max(1, total_turns)

    # 6. Tool-Mode Detection
    reads = len([c for c, a in all_calls if 'read' in c.lower() or 'cat' in str(a).lower()])
    runs = len([c for c, a in all_calls if 'run' in c.lower() or 'write' in c.lower()])
    scout_ratio = reads / max(1, runs)

    print(f"Peer Mentions: {len(peer_mentions)}")
    print(f"Compliance:    {comp_count}")
    print(f"Independence:  {ind_count}")
    print(f"TINGLE SCORE:  {tingle:.2f}")
    print(f"LOOP RATIO:    {loop_ratio:.2f}")
    print(f"PREACHING:     {preaching_index:.2f} (Index)")
    print(f"SCOUT RATIO:   {scout_ratio:.2f} (Read/Run)")
    
    if preaching_index > 0.8:
        print("BEHAVIOR:      DOMINATING PREACHER")
    elif preaching_index < 0.2:
        print("BEHAVIOR:      INCUBATING SCOUT")

    if scout_ratio > 1.5:
        print("TOOL-MODE:     VALIDATING")
    else:
        print("TOOL-MODE:     GENERATIVE")

    if loop_ratio > 0.4:
        print("WARNING: AGENT IS SPINNING (High Stupidity/Waste)")

if __name__ == "__main__":
    log_path = Path(sys.argv[1])
    bb_path = Path(sys.argv[2])
    analyze_cognition(log_path, bb_path)

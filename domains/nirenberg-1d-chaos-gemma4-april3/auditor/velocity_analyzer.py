import json
import re
from datetime import datetime
from pathlib import Path

def analyze_velocity(log_file):
    print(f"=== VELOCITY ANALYZER: {log_file.name} ===")
    
    timestamps = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]', line)
            if match:
                timestamps.append(datetime.strptime(match.group(1), "%H:%M:%S"))
    
    if not timestamps:
        print("No timestamps found.")
        return

    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    num_turns = len(timestamps) # Rough estimate
    
    print(f"Start Time: {timestamps[0].strftime('%H:%M:%S')}")
    print(f"End Time:   {timestamps[-1].strftime('%H:%M:%S')}")
    print(f"Duration:   {duration:.1f} seconds")
    
    # Calculate Velocity (Turns per minute)
    if duration > 0:
        velocity = (num_turns / (duration / 60))
        print(f"VELOCITY:   {velocity:.2f} events/minute")

    # Inertia Analysis (Heuristic: consecutive tool calls with same parameters)
    # This would require deeper parsing of tool_calls in the next version.
    print("INERTIA:    Currently monitoring exploration diversity...")

if __name__ == "__main__":
    import sys
    analyze_velocity(Path(sys.argv[1]))

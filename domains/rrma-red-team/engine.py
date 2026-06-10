#!/usr/bin/env python3
"""
engine.py — rrma-red-team harness stub.

Agents don't run this directly. They:
1. Write a TokenOptimizer subclass in ~/claudini/claudini/methods/rrma/vN/
2. Run the Claudini benchmark: bash run.sh <method_name>
3. Parse SCORE= from output, append to results.tsv

This file exists so the outer-loop scaffold is satisfied.
"""
import subprocess, sys, re
from pathlib import Path

if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "gcg"
    domain = Path(__file__).parent
    result = subprocess.run(
        ["bash", str(domain / "run.sh"), method],
        capture_output=True, text=True
    )
    print(result.stdout)
    m = re.search(r"SCORE=([0-9.]+)", result.stdout)
    print(m.group(1) if m else "0.0")

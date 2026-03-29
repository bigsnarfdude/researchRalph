#!/usr/bin/env python3
"""
verify_filter.py — run lean_verify on generated proof candidates, keep what compiles.

Usage:
  python3 verify_filter.py \
    --candidates data/candidates_stage1.jsonl \
    --lean-project ~/miniF2F-lean4 \
    --output data/verified_stage1.jsonl \
    --workers 4

Input JSONL fields: problem_id, full_lean (the candidate proof to verify)
Output JSONL: same fields + verified=True, compile_time_s
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def verify_one(item: dict, lean_project: Path, timeout: int = 60) -> dict:
    """Write candidate to temp file and verify with lake env lean."""
    lean_text = item.get("full_lean", "")
    if not lean_text:
        return {**item, "verified": False, "error": "no full_lean field"}

    with tempfile.NamedTemporaryFile(
        suffix=".lean", dir=lean_project, delete=False, mode='w'
    ) as f:
        f.write(lean_text)
        tmp_path = f.name

    start = time.time()
    try:
        result = subprocess.run(
            ["lake", "env", "lean", tmp_path],
            capture_output=True, text=True,
            timeout=timeout, cwd=lean_project,
        )
        elapsed = time.time() - start
        verified = result.returncode == 0
        error = result.stderr[:500] if not verified else None
        return {**item, "verified": verified, "compile_time_s": round(elapsed, 2), "error": error}
    except subprocess.TimeoutExpired:
        return {**item, "verified": False, "error": "timeout", "compile_time_s": timeout}
    except Exception as e:
        return {**item, "verified": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--lean-project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    lean_project = Path(args.lean_project).expanduser()
    candidates = []
    with open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Verifying {len(candidates)} candidates with {args.workers} workers...", file=sys.stderr)

    total = verified_count = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w') as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(verify_one, item, lean_project, args.timeout): item
                for item in candidates
            }
            for future in as_completed(futures):
                total += 1
                result = future.result()
                if result.get("verified"):
                    verified_count += 1
                    out.write(json.dumps(result) + '\n')
                if total % 20 == 0:
                    print(f"  {total}/{len(candidates)} done, {verified_count} verified", file=sys.stderr)

    print(f"\nDone: {verified_count}/{total} verified", file=sys.stderr)
    print(f"Pass rate: {verified_count/total*100:.1f}%", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
score_attempts.py — score ALL lean attempt files by closeness to correct

Runs lake verify on every attempt, captures error output, scores by proximity:

  5 = verified (passes)
  4 = unsolved goals only (right structure, just needs better tactic)
  3 = type mismatch (right tactic, wrong types/arguments)
  2 = unknown identifier/declaration (missing lemma name, close)
  1 = tactic failed / simp failed (wrong tactic family)
  0 = parse error / syntax error (broken structure)

Output JSONL: {problem_id, file, score, error_type, error_snippet, lean_code}

Usage:
  python3 bootstrap/score_attempts.py \
    --attempts ~/researchRalph/domains/rrma-lean/attempts \
    --minif2f ~/miniF2F-lean4 \
    --output ~/data/scored_attempts.jsonl \
    --workers 16
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Error classification ──────────────────────────────────────────────────────

def classify_error(stderr: str, passed: bool) -> tuple[int, str, str]:
    """Returns (score 0-5, error_type, error_snippet)."""
    if passed:
        return 5, "verified", ""

    # Extract first error line
    lines = stderr.strip().splitlines()
    error_lines = [l for l in lines if "error:" in l or "warning:" in l]
    snippet = error_lines[0][:120] if error_lines else lines[0][:120] if lines else ""

    text = stderr.lower()

    # Score 4: unsolved goals — right proof structure, just incomplete
    if "unsolved goals" in text:
        return 4, "unsolved_goals", snippet

    # Score 3: type mismatch — right tactic, wrong types
    if "type mismatch" in text or "application type mismatch" in text:
        return 3, "type_mismatch", snippet

    # Score 2: unknown identifier/declaration — missing lemma, close
    if "unknown identifier" in text or "unknown declaration" in text or "unknown tactic" in text:
        return 2, "unknown_identifier", snippet

    # Score 2: function expected — close, structural issue
    if "function expected" in text:
        return 2, "function_expected", snippet

    # Score 1: tactic failed — wrong tactic family
    if "tactic" in text and "failed" in text:
        return 1, "tactic_failed", snippet

    if "simp made no progress" in text or "omega could not" in text:
        return 1, "tactic_no_progress", snippet

    if "linarith failed" in text or "ring failed" in text or "norm_num failed" in text:
        return 1, "tactic_failed", snippet

    # Score 0: parse/syntax error
    if "expected" in text or "unexpected token" in text or "parsing" in text:
        return 0, "syntax_error", snippet

    # Default: unknown failure
    return 1, "unknown_failure", snippet


# ── Verify single file ────────────────────────────────────────────────────────

def verify_and_score(lean_file: Path, minif2f: Path, timeout: int = 45) -> dict:
    try:
        result = subprocess.run(
            ["bash", "-c", f"source ~/.elan/env && cd '{minif2f}' && lake env lean '{lean_file}'"],
            capture_output=True, text=True, timeout=timeout
        )
        passed = result.returncode == 0
        stderr = result.stderr + result.stdout
    except subprocess.TimeoutExpired:
        return {
            "problem_id": lean_file.stem,
            "file": str(lean_file),
            "score": 1,
            "error_type": "timeout",
            "error_snippet": "timeout after 45s",
            "passed": False,
        }
    except Exception as e:
        return {
            "problem_id": lean_file.stem,
            "file": str(lean_file),
            "score": 0,
            "error_type": "exception",
            "error_snippet": str(e)[:100],
            "passed": False,
        }

    score, error_type, snippet = classify_error(stderr, passed)
    return {
        "problem_id": lean_file.stem,
        "file": str(lean_file),
        "score": score,
        "error_type": error_type,
        "error_snippet": snippet,
        "passed": passed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--minif2f", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-score", type=int, default=0,
                        help="Only output attempts with score >= this (default: 0 = all)")
    args = parser.parse_args()

    attempts_dir = Path(args.attempts).expanduser()
    minif2f = Path(args.minif2f).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(attempts_dir.rglob("*.lean"))
    print(f"Scoring {len(files)} attempt files with {args.workers} workers...")

    # Resume
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["file"])
        print(f"Resuming: {len(done)} already done")

    files = [f for f in files if str(f) not in done]
    print(f"Remaining: {len(files)}")

    score_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total = 0

    with out_path.open("a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(verify_and_score, f, minif2f): f for f in files}
            for i, fut in enumerate(as_completed(futures)):
                result = fut.result()
                score_counts[result["score"]] += 1
                total += 1
                if result["score"] >= args.min_score:
                    out_f.write(json.dumps(result) + "\n")
                if total % 500 == 0:
                    out_f.flush()
                    print(f"  {total}/{len(files)} | "
                          f"verified={score_counts[5]} "
                          f"unsolved_goals={score_counts[4]} "
                          f"type_mismatch={score_counts[3]} "
                          f"unknown_id={score_counts[2]} "
                          f"tactic_fail={score_counts[1]} "
                          f"syntax={score_counts[0]}")

    print(f"\n=== Final Scores ===")
    labels = {5: "verified      ", 4: "unsolved_goals", 3: "type_mismatch ",
              2: "unknown_id    ", 1: "tactic_failed ", 0: "syntax_error  "}
    for s in range(5, -1, -1):
        pct = score_counts[s] / total * 100 if total else 0
        print(f"  {labels[s]}: {score_counts[s]:5d} ({pct:.1f}%)")
    print(f"  Total: {total}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()

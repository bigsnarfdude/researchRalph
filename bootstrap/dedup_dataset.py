#!/usr/bin/env python3
"""
dedup_dataset.py — Deduplicate dataset to 1 best proof per problem

Selection priority:
  1. No sorry (hard filter — sorry proofs never win over non-sorry)
  2. Highest score (5 > 4 > 3 > ...)
  3. Has thinking trace (gold/silver preferred)
  4. Tactic style preference: reasoning > simple > shotgun
  5. Shortest proof (Occam's razor — simpler verified proof is better)

Usage:
  python3 bootstrap/dedup_dataset.py \
    --input ~/data/rrma_lean_dataset.jsonl \
    --output ~/data/rrma_lean_dataset_deduped.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


STYLE_RANK = {"reasoning": 3, "simple": 2, "shotgun": 1, "unknown": 0}


def sort_key(record):
    """Higher = better. Returns tuple for sorting."""
    has_sorry = "sorry" in record.get("lean_code", "")
    has_trace = bool(record.get("thinking_trace"))
    style = STYLE_RANK.get(record.get("tactic_style", "unknown"), 0)
    score = record.get("score", 0)
    # Negative proof_chars: shorter is better (among equal-quality proofs)
    proof_chars = record.get("proof_chars", 99999)

    return (
        0 if has_sorry else 1,   # no sorry first
        score,                    # highest score
        1 if has_trace else 0,    # prefer traces
        style,                    # reasoning > simple > shotgun
        -proof_chars,             # shorter proof wins
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Full dataset JSONL")
    parser.add_argument("--output", required=True, help="Deduped dataset JSONL (1 per problem)")
    args = parser.parse_args()

    # Load all records grouped by problem
    by_problem = defaultdict(list)
    total = 0
    sorry_count = 0

    with open(Path(args.input).expanduser()) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if "sorry" in rec.get("lean_code", ""):
                sorry_count += 1
            by_problem[rec["problem_id"]].append(rec)

    print(f"Loaded {total} records across {len(by_problem)} problems")
    print(f"  Sorry proofs: {sorry_count}")

    # Pick best per problem
    deduped = []
    sorry_only_problems = []

    for pid, records in sorted(by_problem.items()):
        best = max(records, key=sort_key)

        # Flag if the best we have still contains sorry
        if "sorry" in best.get("lean_code", ""):
            sorry_only_problems.append(pid)

        deduped.append(best)

    # Write output
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in deduped:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDeduped: {len(deduped)} records (1 per problem)")

    # Stats
    from collections import Counter
    tiers = Counter(r["tier"] for r in deduped)
    styles = Counter(r["tactic_style"] for r in deduped)
    scores = Counter(r["score"] for r in deduped)
    has_trace = sum(1 for r in deduped if r.get("thinking_trace"))
    verified = sum(1 for r in deduped if r["score"] == 5)

    print(f"  Verified (score=5): {verified}")
    print(f"  With trace: {has_trace}")

    print(f"\nBy tier:")
    for t in ["gold", "silver", "verified", "traced", "near_miss", "attempt"]:
        if tiers[t]:
            print(f"  {t:<12}: {tiers[t]:>4}")

    print(f"\nBy tactic style:")
    for s, c in styles.most_common():
        print(f"  {s:<12}: {c:>4}")

    print(f"\nBy score:")
    labels = {5: "verified", 4: "unsolved_goals", 3: "type_mismatch",
              2: "unknown_id", 1: "tactic_failed", 0: "syntax_error"}
    for s in range(5, -1, -1):
        if scores[s]:
            print(f"  {labels[s]:<16}: {scores[s]:>4}")

    if sorry_only_problems:
        print(f"\nWARNING: {len(sorry_only_problems)} problems have ONLY sorry proofs:")
        for p in sorry_only_problems:
            print(f"  {p}")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()

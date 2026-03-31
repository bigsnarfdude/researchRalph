#!/usr/bin/env python3
"""
build_dataset.py — join scored attempts + thinking traces into one publishable dataset

Reads:
  1. scored_attempts.jsonl  — lake verify scores (problem_id, file, score, passed)
  2. rrma_lean_traces.jsonl — thinking traces with <think> chains (messages, metadata)
  3. The .lean files themselves — for the actual proof code

Writes:
  rrma_lean_dataset.jsonl — one unified JSONL with schema:
    {
      "problem_id":       "mathd_algebra_77",
      "lean_code":        "import Mathlib\n...",
      "score":            5,
      "passed":           true,
      "error_type":       "verified",
      "experiment":       "exp022",
      "thinking_trace":   "<think>Let me analyze...</think>" or null,
      "tactic_style":     "reasoning" | "shotgun" | "unknown",
      "thinking_chars":   47575 or 0,
      "proof_chars":      785,
      "source":           "rrma-lean-v4"
    }

Usage:
  python3 bootstrap/build_dataset.py \
    --scored ~/data/scored_attempts.jsonl \
    --traces ~/data/rrma_lean_traces.jsonl \
    --output ~/data/rrma_lean_dataset.jsonl \
    --min-score 0
"""

import argparse
import json
import re
from pathlib import Path


def classify_tactic_style(lean_code: str) -> str:
    """Classify proof as reasoning, shotgun, simple, or unknown."""
    if not lean_code:
        return "unknown"

    # Extract proof body only (after := by)
    proof_start = lean_code.find(":= by")
    proof = lean_code[proof_start:] if proof_start >= 0 else lean_code

    # Count non-comment, non-blank tactic lines (used for simple check)
    tactic_lines = [l.strip() for l in proof.splitlines()
                    if l.strip()
                    and not l.strip().startswith("--")
                    and not l.strip().startswith(":= by")
                    and not l.strip().startswith("import")
                    and not l.strip().startswith("open ")
                    and not l.strip().startswith("set_option")
                    and not l.strip().startswith("theorem")
                    and not l.strip().startswith("noncomputable")]

    # Simple FIRST: ≤3 tactic lines is simple, period.
    # This prevents short proofs with `have` from being called "reasoning".
    if len(tactic_lines) <= 3:
        return "simple"

    # Shotgun: explicit `| solve |` chains
    if proof.count("| solve |") >= 3:
        return "shotgun"

    # Shotgun: `first` block with many tactic alternatives
    # Only count pipes inside `first` blocks, not Lean 4 match arms
    first_blocks = re.findall(r'first\s*\n?((?:\s*\|.*\n?)+)', proof)
    if first_blocks:
        total_alts = sum(block.count("|") for block in first_blocks)
        if total_alts >= 5:
            return "shotgun"

    # Shotgun: multiple `first` keywords (even without `solve`)
    if proof.count("first") >= 3:
        return "shotgun"

    # Reasoning: structured proof tactics (multi-step)
    reasoning_markers = ["rcases", "have ", "calc", "obtain", "suffices",
                         "by_contra", "induction", "cases ", "match ",
                         "strongRecOn", "Nat.strongRecOn", "convert "]
    if any(m in proof for m in reasoning_markers):
        return "reasoning"

    return "simple"


def extract_thinking(messages: list) -> str | None:
    """Extract <think>...</think> from assistant message."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if m:
                return m.group(0)
    return None


def trace_is_relevant(thinking: str, problem_id: str, all_problem_ids: set) -> bool:
    """Check if a thinking trace is actually about this problem, not polluted.

    Rejects traces where:
      1. The target problem is never mentioned (name or short variant)
      2. Other problems are mentioned MORE than the target (contamination)
    """
    if not thinking:
        return False

    think_lower = thinking.lower()

    # Count mentions of target problem
    pid_lower = problem_id.lower()
    target_mentions = think_lower.count(pid_lower)

    # Also check short variant (e.g. "algebra_77" from "mathd_algebra_77")
    parts = problem_id.split("_")
    if len(parts) >= 2:
        short = "_".join(parts[-2:]).lower()
        target_mentions += think_lower.count(short)

    # Must mention the target at least once
    if target_mentions == 0:
        return False

    # Count mentions of OTHER problems — if others dominate, it's polluted
    other_mentions = 0
    for other_pid in all_problem_ids:
        if other_pid == problem_id:
            continue
        other_lower = other_pid.lower()
        cnt = think_lower.count(other_lower)
        if cnt > 0:
            other_mentions += cnt

    # Polluted if other problems are mentioned 3x more than target
    if other_mentions > target_mentions * 3 and other_mentions > 5:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", required=True, help="scored_attempts.jsonl")
    parser.add_argument("--traces", required=True, help="rrma_lean_traces.jsonl")
    parser.add_argument("--output", required=True, help="Output dataset JSONL")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Only include attempts with score >= this")
    parser.add_argument("--verified-only", action="store_true",
                        help="Only include verified (score=5) attempts")
    args = parser.parse_args()

    # Load traces indexed by problem
    traces_by_problem = {}
    traces_path = Path(args.traces).expanduser()
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec["metadata"]["problem"]
                thinking = extract_thinking(rec["messages"])
                if thinking:
                    traces_by_problem[pid] = {
                        "thinking_trace": thinking,
                        "thinking_chars": rec["metadata"].get("thinking_chars", len(thinking)),
                    }
        print(f"Loaded {len(traces_by_problem)} thinking traces")

    # Collect all problem IDs for contamination check
    scored_path = Path(args.scored).expanduser()
    all_problem_ids = set()
    with open(scored_path) as f:
        for line in f:
            if line.strip():
                all_problem_ids.add(json.loads(line)["problem_id"])
    print(f"Found {len(all_problem_ids)} unique problem IDs")

    # Filter traces for relevance (reject polluted traces)
    filtered_traces = {}
    dropped_traces = 0
    for pid, tinfo in traces_by_problem.items():
        trace_text = tinfo.get("thinking_trace", "")
        if trace_is_relevant(trace_text, pid, all_problem_ids):
            filtered_traces[pid] = tinfo
        else:
            dropped_traces += 1
    print(f"Traces after relevance filter: {len(filtered_traces)} (dropped {dropped_traces} polluted)")
    traces_by_problem = filtered_traces

    # Process scored attempts
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    failed_reads = 0

    with open(out_path, "w") as out_f:
        with open(scored_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total += 1

                score = rec["score"]
                if args.verified_only and score != 5:
                    continue
                if score < args.min_score:
                    continue

                # Read the actual lean file
                lean_file = Path(rec["file"])
                if not lean_file.exists():
                    failed_reads += 1
                    continue
                lean_code = lean_file.read_text(errors="replace")

                # Sorry filter: if proof contains sorry, it cannot be verified
                has_sorry = "sorry" in lean_code
                if has_sorry and score == 5:
                    score = 4  # Downgrade: sorry ≠ verified
                    rec["score"] = 4
                if has_sorry:
                    rec["passed"] = False

                # Extract experiment name from path
                parts = rec["file"].split("/")
                exp_parts = [p for p in parts if p.startswith("exp") or p.startswith("smoke") or p.startswith("_tmp")]
                experiment = exp_parts[0] if exp_parts else "unknown"

                # Get thinking trace if available
                pid = rec["problem_id"]
                trace_info = traces_by_problem.get(pid, {})

                # Count proof lines (after theorem statement)
                proof_start = lean_code.find(":= by")
                proof_code = lean_code[proof_start:] if proof_start >= 0 else lean_code
                proof_chars = len(proof_code)

                # Assign tier
                has_trace = bool(trace_info.get("thinking_trace"))
                if has_trace and score == 5:
                    tier = "gold"       # verified proof + Opus reasoning
                elif has_trace and score == 4:
                    tier = "silver"     # near-miss proof + Opus reasoning
                elif score == 5:
                    tier = "verified"   # correct proof, no reasoning trace
                elif has_trace:
                    tier = "traced"     # reasoning trace, proof didn't land
                elif score == 4:
                    tier = "near_miss"  # close attempt, no trace
                else:
                    tier = "attempt"    # other attempts

                row = {
                    "problem_id": pid,
                    "lean_code": lean_code,
                    "score": score,
                    "passed": rec.get("passed", score == 5),
                    "error_type": rec.get("error_type", ""),
                    "tier": tier,
                    "experiment": experiment,
                    "thinking_trace": trace_info.get("thinking_trace"),
                    "tactic_style": classify_tactic_style(lean_code),
                    "thinking_chars": trace_info.get("thinking_chars", 0),
                    "proof_chars": proof_chars,
                    "source": "rrma-lean-v4",
                    "model": "claude-opus-4-6",
                }
                out_f.write(json.dumps(row) + "\n")
                written += 1

                if written % 2000 == 0:
                    print(f"  {written} written ({total} processed)")

    print(f"\nDone: {written} records written ({total} processed, {failed_reads} unreadable files)")
    print(f"Output: {out_path}")

    # Summary stats
    with open(out_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    from collections import Counter
    scores = Counter(r["score"] for r in records)
    styles = Counter(r["tactic_style"] for r in records)
    tiers = Counter(r["tier"] for r in records)
    has_trace = sum(1 for r in records if r["thinking_trace"])
    problems = len(set(r["problem_id"] for r in records))
    verified_problems = len(set(r["problem_id"] for r in records if r["passed"]))

    print(f"\n=== Dataset Summary ===")
    print(f"Records:            {len(records)}")
    print(f"Unique problems:    {problems}")
    print(f"Verified problems:  {verified_problems}")
    print(f"With thinking trace: {has_trace}")
    print(f"\nBy tier:")
    tier_order = ["gold", "silver", "verified", "traced", "near_miss", "attempt"]
    tier_desc = {
        "gold": "verified + Opus reasoning",
        "silver": "near-miss + Opus reasoning",
        "verified": "correct proof, no trace",
        "traced": "Opus reasoning, proof failed",
        "near_miss": "close attempt, no trace",
        "attempt": "other attempts",
    }
    for t in tier_order:
        if tiers[t]:
            print(f"  {t:<12}: {tiers[t]:>6}  ({tier_desc[t]})")
    print(f"\nBy score:")
    labels = {5: "verified", 4: "unsolved_goals", 3: "type_mismatch",
              2: "unknown_id", 1: "tactic_failed", 0: "syntax_error"}
    for s in range(5, -1, -1):
        if scores[s]:
            print(f"  {labels[s]:<16}: {scores[s]:>6}")
    print(f"\nBy tactic style:")
    for style, count in styles.most_common():
        print(f"  {style:<16}: {count:>6}")


if __name__ == "__main__":
    main()

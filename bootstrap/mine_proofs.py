#!/usr/bin/env python3
"""
mine_proofs.py — extract verified Lean proofs from rrma-lean attempts/

Usage:
  python3 mine_proofs.py --attempts-dir ../domains/rrma-lean/attempts \
                          --output ../data/seed_proofs.jsonl

Output: JSONL, one proof per line:
  {problem_id, theorem_statement, proof_body, full_lean,
   tactic_count, verified, source, tactic_types}
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


TRIVIAL_PROOFS = {"by decide", "by exact?", "by simp", "by tauto", "by trivial"}
NONTRIVIAL_TACTICS = {
    "ring", "linarith", "nlinarith", "omega", "norm_num",
    "field_simp", "linear_combination", "push_cast", "zify",
    "rcases", "cases", "induction", "apply", "exact",
    "have", "obtain", "use", "constructor", "ext",
    "simp only", "rw", "calc",
}


def extract_theorem_and_proof(lean_text: str) -> tuple[str, str] | None:
    """Extract theorem statement and proof body from a .lean file."""
    # Find 'theorem ... := by' or 'theorem ... where'
    m = re.search(r'(theorem\s+\w+.*?):=\s*by\b', lean_text, re.DOTALL)
    if not m:
        return None
    stmt = m.group(1).strip()
    proof_start = m.end()
    proof_body = lean_text[proof_start:].strip()
    return stmt, proof_body


def count_tactics(proof_body: str) -> int:
    """Count non-trivial tactic lines."""
    lines = [l.strip() for l in proof_body.split('\n') if l.strip()]
    return len(lines)


def get_tactic_types(proof_body: str) -> list[str]:
    """Return which known tactics appear in the proof."""
    found = []
    lower = proof_body.lower()
    for t in NONTRIVIAL_TACTICS:
        if t in lower:
            found.append(t)
    return sorted(found)


def is_nontrivial(proof_body: str) -> bool:
    """Reject trivially short or sorry-containing proofs."""
    stripped = proof_body.strip()
    if 'sorry' in stripped:
        return False
    if stripped in TRIVIAL_PROOFS:
        return False
    tactics = get_tactic_types(proof_body)
    return len(tactics) >= 1


def verify_lean(lean_file: Path, timeout: int = 120) -> bool:
    """Run lake env lean on the file. Returns True if it compiles."""
    try:
        result = subprocess.run(
            ['lake', 'env', 'lean', str(lean_file)],
            capture_output=True, text=True, timeout=timeout,
            cwd=lean_file.parent,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def process_file(lean_file: Path, source: str, verify: bool) -> dict | None:
    """Process one .lean file into a training example."""
    text = lean_file.read_text(encoding='utf-8', errors='ignore')

    result = extract_theorem_and_proof(text)
    if result is None:
        return None
    stmt, proof_body = result

    if not is_nontrivial(proof_body):
        return None

    verified = verify_lean(lean_file) if verify else None

    return {
        "problem_id": lean_file.stem,
        "theorem_statement": stmt,
        "proof_body": proof_body,
        "full_lean": text,
        "tactic_count": count_tactics(proof_body),
        "tactic_types": get_tactic_types(proof_body),
        "verified": verified,
        "source": source,
        "file": str(lean_file),
    }


def format_sft_example(example: dict) -> dict:
    """Format as SFT training pair for Qwen chat format."""
    prompt = (
        "Prove the following theorem in Lean 4 using Mathlib:\n\n"
        f"```lean\n{example['full_lean'][:example['full_lean'].index(':= by')].strip()}\n```"
    )
    response = f"```lean\n{example['full_lean']}\n```"
    return {
        **example,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "system", "content": "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Mine verified Lean proofs from rrma-lean attempts")
    parser.add_argument("--attempts-dir", required=True, help="Path to attempts/ directory")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--verify", action="store_true", help="Re-verify each proof with lean_verify")
    parser.add_argument("--min-tactics", type=int, default=1, help="Min tactic count to include")
    parser.add_argument("--sft-format", action="store_true", help="Include SFT chat format in output")
    parser.add_argument("--multi-proof", action="store_true", help="Include multiple distinct proofs per problem (up to 3)")
    args = parser.parse_args()

    attempts_dir = Path(args.attempts_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lean_files = list(attempts_dir.rglob("*.lean"))
    print(f"Found {len(lean_files)} .lean files in {attempts_dir}", file=sys.stderr)

    total = verified = skipped = 0
    # multi-proof: track (problem_id -> set of proof fingerprints)
    seen_ids: dict = {}  # problem_id -> count of proofs kept

    with output_path.open('w') as out:
        for lean_file in sorted(lean_files):
            # Source = experiment name (parent dir or grandparent)
            source = lean_file.parent.name
            if source == "attempts":
                source = lean_file.parent.parent.name

            total += 1
            example = process_file(lean_file, source, args.verify)
            if example is None:
                skipped += 1
                continue

            pid = example['problem_id']
            proof_fp = example['proof_body'][:120]  # fingerprint

            if args.multi_proof:
                # Keep up to 3 distinct proofs per problem
                if pid not in seen_ids:
                    seen_ids[pid] = set()
                if proof_fp in seen_ids[pid] or len(seen_ids[pid]) >= 3:
                    skipped += 1
                    continue
                seen_ids[pid].add(proof_fp)
                # Tag with proof variant number
                example['problem_id'] = f"{pid}_v{len(seen_ids[pid])}"
            else:
                if pid in seen_ids:
                    skipped += 1
                    continue
                seen_ids[pid] = {proof_fp}

            if example['tactic_count'] < args.min_tactics:
                skipped += 1
                continue

            if args.sft_format:
                example = format_sft_example(example)

            if args.verify and example.get('verified') is False:
                print(f"  FAIL: {lean_file.name}", file=sys.stderr)
                skipped += 1
                continue

            out.write(json.dumps(example) + '\n')
            verified += 1

            if verified % 10 == 0:
                print(f"  {verified} proofs extracted...", file=sys.stderr)

    print(f"\nDone: {verified} proofs extracted, {skipped} skipped of {total} total", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

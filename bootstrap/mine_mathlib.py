#!/usr/bin/env python3
"""
mine_mathlib.py — extract tactic proofs from Mathlib4 source files.

No compilation needed — parses .lean source directly.
Filters to proofs using competition-math tactics.

Usage:
  python3 mine_mathlib.py \
    --mathlib-dir ~/mathlib4 \
    --output data/mathlib_filtered.jsonl \
    --dirs Mathlib/Algebra Mathlib/NumberTheory \
    --sft-format
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Tactics that appear in MiniF2F solutions — filter Mathlib to match
TARGET_TACTICS = {
    "linarith", "nlinarith", "omega", "norm_num", "ring",
    "field_simp", "linear_combination", "push_cast", "zify",
    "positivity", "norm_cast", "exact?", "apply?",
}

# Tactics that signal overly complex proofs — avoid
COMPLEX_TACTICS = {
    "CategoryTheory", "AlgebraicGeometry", "RepresentationTheory",
    "Topology.Sheaf", "sorry",
}

# Max proof lines (keep short, MiniF2F-like proofs)
MAX_PROOF_LINES = 30
MIN_PROOF_LINES = 1


def extract_theorems(lean_text: str, source_file: str) -> list[dict]:
    """
    Extract all theorem/lemma declarations with tactic proofs from a .lean file.
    Returns list of dicts with decl_name, statement, proof_body, full_text.
    """
    results = []

    # Match: theorem/lemma NAME ... := by PROOF
    # Handle multiline statements
    pattern = re.compile(
        r'^((?:theorem|lemma)\s+(\w+)[^:=]*):=\s*by\b',
        re.MULTILINE
    )

    for m in pattern.finditer(lean_text):
        decl_start = m.start()
        stmt = m.group(1).strip()
        name = m.group(2)
        proof_start = m.end()

        # Extract proof body: everything after 'by' until next top-level decl
        rest = lean_text[proof_start:]
        # Next theorem/lemma/def at column 0
        next_decl = re.search(r'\n(?=theorem |lemma |def |noncomputable def |private |@\[)', rest)
        if next_decl:
            proof_body = rest[:next_decl.start()].strip()
        else:
            proof_body = rest[:2000].strip()  # cap at 2000 chars

        if not proof_body:
            continue

        proof_lines = [l for l in proof_body.split('\n') if l.strip()]
        if not (MIN_PROOF_LINES <= len(proof_lines) <= MAX_PROOF_LINES):
            continue

        # Must contain at least one target tactic
        proof_lower = proof_body.lower()
        if not any(t in proof_lower for t in TARGET_TACTICS):
            continue

        # Reject if contains complex/irrelevant tactics
        full_lower = lean_text[:proof_start + len(proof_body)].lower()
        if any(c.lower() in full_lower for c in COMPLEX_TACTICS):
            continue

        # Build the full lean snippet (with imports)
        full_lean = (
            "import Mathlib\n"
            "set_option maxHeartbeats 400000\n"
            "open BigOperators Real Nat Topology Rat\n\n"
            f"{stmt} := by\n{proof_body}"
        )

        results.append({
            "decl_name": name,
            "theorem_statement": stmt,
            "proof_body": proof_body,
            "full_lean": full_lean,
            "proof_lines": len(proof_lines),
            "tactic_types": sorted(t for t in TARGET_TACTICS if t in proof_lower),
            "source_file": source_file,
            "verified": None,  # not re-verified, trusted from Mathlib CI
        })

    return results


def format_sft_example(example: dict) -> dict:
    stmt = example["theorem_statement"]
    full = example["full_lean"]
    prompt = f"Prove the following theorem in Lean 4 using Mathlib:\n\n```lean\n{stmt}\n```"
    response = f"```lean\n{full}\n```"
    return {
        **example,
        "prompt": prompt,
        "response": response,
        "messages": [
            {"role": "system", "content": "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dirs", nargs="+",
                        default=["Mathlib/Algebra", "Mathlib/NumberTheory"])
    parser.add_argument("--sft-format", action="store_true")
    parser.add_argument("--max-per-file", type=int, default=50)
    args = parser.parse_args()

    mathlib = Path(args.mathlib_dir).expanduser()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lean_files = []
    for d in args.dirs:
        target = mathlib / d
        if target.exists():
            lean_files.extend(sorted(target.rglob("*.lean")))
        else:
            print(f"  Warning: {target} not found", file=sys.stderr)

    print(f"Scanning {len(lean_files)} .lean files in {args.dirs}...", file=sys.stderr)

    seen_names = set()
    total = 0

    with out_path.open("w") as out:
        for lean_file in lean_files:
            try:
                text = lean_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel = str(lean_file.relative_to(mathlib))
            examples = extract_theorems(text, rel)

            # Cap per file to avoid over-representing one module
            examples = examples[:args.max_per_file]

            for ex in examples:
                if ex["decl_name"] in seen_names:
                    continue
                seen_names.add(ex["decl_name"])

                if args.sft_format:
                    ex = format_sft_example(ex)

                out.write(json.dumps(ex) + "\n")
                total += 1

            if total % 500 == 0 and total > 0:
                print(f"  {total} extracted...", file=sys.stderr)

    print(f"\nDone: {total} theorems from Mathlib", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

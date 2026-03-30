#!/usr/bin/env python3
"""
generate_opus_traces.py — generate Lean 4 proof traces using Claude Opus with extended thinking

The dataset we're building: Opus CoT traces (thinking + proof) on MiniF2F problems.
These become SFT data for distilling theorem-proving ability into smaller models.

Output format (DeepSeek-R1 style):
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Prove the following theorem in Lean 4:\n\n```lean\n{stmt}\n```"},
    {"role": "assistant", "content": "<think>\n{thinking}\n</think>\n\n```lean\n{proof}\n```"}
  ], "metadata": {"problem": "...", "thinking_tokens": N, "verified": true/false}}

Usage:
  export ANTHROPIC_API_KEY=...
  python3 bootstrap/generate_opus_traces.py \
    --minif2f ~/miniF2F-lean4 \
    --output data/opus_traces.jsonl \
    [--n-problems 50] \
    [--thinking-budget 8000] \
    [--verify] \
    [--lean-project ~/miniF2F-lean4]

Options:
  --minif2f PATH       MiniF2F-lean4 repo (reads .lean files for problem statements)
  --output FILE        Output JSONL (default: data/opus_traces.jsonl); resumes if exists
  --n-problems N       Max problems to process (default: all 244)
  --thinking-budget N  Max thinking tokens per problem (default: 8000, min 1024)
  --verify             Run lake env lean to verify each proof (requires lake)
  --lean-project PATH  miniF2F-lean4 for lake verification (default: same as --minif2f)
  --split valid|test   Which MiniF2F split to use (default: valid)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover using Mathlib. "
    "Given a theorem statement, think carefully about the proof strategy, "
    "then write a complete Lean 4 proof that compiles without sorry."
)

LEAN_HEADER = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n\n"


def load_problems(minif2f_dir: Path, split: str = "valid") -> list[dict]:
    """Load theorem statements from miniF2F-lean4 Valid/ or Test/ directory."""
    split_dir = minif2f_dir / "MiniF2F" / split.capitalize()
    if not split_dir.exists():
        # Try lowercase
        split_dir = minif2f_dir / "MiniF2F" / split.lower()
    if not split_dir.exists():
        print(f"ERROR: cannot find {split_dir}", file=sys.stderr)
        sys.exit(1)

    problems = []
    for lean_file in sorted(split_dir.glob("*.lean")):
        problem_name = lean_file.stem
        text = lean_file.read_text(errors="replace")

        # Extract the theorem statement (up to := by or :=)
        m = re.search(r'(theorem\s+\w+.*?)(?::=\s*by\b|:=\s*\n)', text, re.DOTALL)
        if m:
            stmt = m.group(1).strip()
        else:
            # Take the whole file as-is (some files have full proofs already)
            stmt = text.strip()

        # Get the sorry-form (full file with sorry as placeholder)
        sorry_proof = LEAN_HEADER + stmt + " := by\n  sorry"

        problems.append({
            "problem_id": problem_name,
            "theorem_statement": stmt,
            "sorry_form": sorry_proof,
            "source_file": str(lean_file),
        })

    return problems


def extract_lean_from_text(text: str, stmt: str) -> str:
    """Extract lean code block from assistant text response."""
    m = re.search(r'```lean\s*(.*?)```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code.startswith("import"):
            return code
        if ":= by" in code or ":=" in code:
            return LEAN_HEADER + code
        # Treat as proof body
        return LEAN_HEADER + stmt + " := by\n" + code
    # No code block — treat whole response as proof body
    return LEAN_HEADER + stmt + " := by\n" + text.strip()


def verify_lean(lean_code: str, problem_name: str, lean_project: Path, timeout: int = 60) -> bool:
    """Write lean_code to a temp file and verify with lake env lean."""
    tmp = Path(f"/tmp/lean_verify_{problem_name}.lean")
    tmp.write_text(lean_code)
    try:
        result = subprocess.run(
            ["bash", "-c", f"cd '{lean_project}' && lake env lean '{tmp}'"],
            capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def call_opus(client, problem: dict, thinking_budget: int) -> dict | None:
    """Call Claude Opus with extended thinking. Returns trace dict or None on failure."""
    stmt = problem["theorem_statement"]
    user_content = (
        f"Prove the following theorem in Lean 4 using Mathlib:\n\n"
        f"```lean\n{stmt}\n```\n\n"
        f"Write the complete Lean 4 proof file (starting with `import Mathlib`)."
    )

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=thinking_budget + 2048,
            thinking={"type": "enabled", "budget_tokens": thinking_budget},
            messages=[{"role": "user", "content": user_content}],
            system=SYSTEM_PROMPT,
        )
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None

    # Extract thinking and text blocks
    thinking_parts = []
    text_parts = []
    for block in response.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)

    thinking = "\n\n".join(thinking_parts)
    text = "\n\n".join(text_parts)

    lean_code = extract_lean_from_text(text, stmt)

    return {
        "thinking": thinking,
        "text": text,
        "lean_code": lean_code,
        "thinking_tokens": sum(len(t.split()) for t in thinking_parts),  # approx
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def to_sft_record(problem: dict, trace: dict, verified: bool) -> dict:
    """Format as DeepSeek-R1 style SFT record."""
    stmt = problem["theorem_statement"]
    user_content = f"Prove the following theorem in Lean 4:\n\n```lean\n{stmt}\n```"
    assistant_content = (
        f"<think>\n{trace['thinking']}\n</think>\n\n"
        f"```lean\n{trace['lean_code']}\n```"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "problem": problem["problem_id"],
            "thinking_tokens": trace["thinking_tokens"],
            "input_tokens": trace["input_tokens"],
            "output_tokens": trace["output_tokens"],
            "verified": verified,
            "proof_chars": len(trace["lean_code"]),
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minif2f", required=True, help="Path to miniF2F-lean4 repo")
    parser.add_argument("--output", default="data/opus_traces.jsonl")
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=8000)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--lean-project", default=None)
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    minif2f_dir = Path(args.minif2f).expanduser()
    lean_project = Path(args.lean_project).expanduser() if args.lean_project else minif2f_dir
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems = load_problems(minif2f_dir, args.split)
    print(f"Loaded {len(problems)} problems from {args.split}", flush=True)

    # Resume: skip already-done problems
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done_ids.add(rec["metadata"]["problem"])
                    except Exception:
                        pass
        print(f"Resuming: {len(done_ids)} already done", flush=True)

    problems = [p for p in problems if p["problem_id"] not in done_ids]
    if args.n_problems:
        problems = problems[:args.n_problems]

    print(f"Generating traces for {len(problems)} problems (thinking_budget={args.thinking_budget})", flush=True)

    total = 0
    verified_count = 0

    with out_path.open("a") as out_f:
        for i, prob in enumerate(problems):
            pid = prob["problem_id"]
            print(f"\n[{i+1}/{len(problems)}] {pid}", flush=True)

            trace = call_opus(client, prob, args.thinking_budget)
            if trace is None:
                print(f"  SKIP (API error)", flush=True)
                time.sleep(5)
                continue

            print(f"  thinking: ~{trace['thinking_tokens']} tokens", flush=True)

            verified = False
            if args.verify:
                verified = verify_lean(trace["lean_code"], pid, lean_project)
                print(f"  lean verify: {'PASS' if verified else 'FAIL'}", flush=True)
                if verified:
                    verified_count += 1

            record = to_sft_record(prob, trace, verified)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            total += 1

            # Rate limit: ~1 req/s to be safe
            time.sleep(1)

    print(f"\n=== Done ===", flush=True)
    print(f"Traces written: {total}", flush=True)
    if args.verify:
        print(f"Verified (pass@1): {verified_count}/{total} = {verified_count/max(total,1):.1%}", flush=True)
    print(f"Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()

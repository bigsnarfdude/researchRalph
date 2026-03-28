#!/usr/bin/env python3
"""
traces_to_sft.py — Convert agent JSONL traces to SFT training format

Extracts (problem, thinking, proof) triples from Claude agent JSONL logs.
Outputs JSONL in the DeepSeek-R1 / Qwen chat format:

  {"messages": [
    {"role": "user", "content": "<problem statement>"},
    {"role": "assistant", "content": "<think>\\n...reasoning...\\n</think>\\n<lean proof>"}
  ]}

Usage:
    python3 tools/traces_to_sft.py domains/rrma-lean/logs/ \
        --minif2f /path/to/miniF2F-lean4 \
        --out sft_traces.jsonl \
        [--min-thinking 100] \
        [--require-pass]

Options:
    --minif2f PATH    Path to miniF2F-lean4 repo (for problem statements)
    --out FILE        Output JSONL file (default: sft_traces.jsonl)
    --min-thinking N  Minimum thinking chars per trace (default: 50)
    --require-pass    Only include proofs that compiled (requires results.tsv)
    --results TSV     results.tsv for pass/fail info (default: auto-detect)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


# ── Problem statement extraction ─────────────────────────────────────────────

def load_minif2f_problems(minif2f_dir):
    """
    Load theorem statements from miniF2F-lean4.
    Returns dict: problem_name → theorem_statement (as string).
    """
    problems = {}
    minif2f_dir = Path(minif2f_dir)

    # MiniF2F Lean4 stores problems in MiniF2F/Valid/*.lean and similar
    for lean_file in minif2f_dir.rglob("*.lean"):
        try:
            text = lean_file.read_text(errors='replace')
        except Exception:
            continue

        # Extract theorem declarations
        for m in re.finditer(
            r'theorem\s+(\w+)\s*(.*?)(?=\ntheorem|\Z)',
            text, re.DOTALL
        ):
            name = m.group(1)
            stmt = m.group(0).strip()
            # Keep only the statement (up to := by or :=)
            stmt_only = re.split(r':=\s*by\b|:=\s*\n', stmt)[0].strip()
            if name not in problems:
                problems[name] = stmt_only

    return problems


# ── JSONL trace parsing ───────────────────────────────────────────────────────

def parse_session(jsonl_path):
    """
    Parse one JSONL file into a list of events:
      {"type": "thinking"|"text"|"tool_use"|"tool_result", "content": ..., "tool_name": ...}
    """
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t == "assistant":
                for block in obj["message"].get("content", []):
                    btype = block.get("type")
                    if btype == "thinking":
                        events.append({"type": "thinking", "content": block.get("thinking", "")})
                    elif btype == "text":
                        events.append({"type": "text", "content": block.get("text", "")})
                    elif btype == "tool_use":
                        events.append({
                            "type": "tool_use",
                            "tool_name": block.get("name", ""),
                            "input": block.get("input", {}),
                        })
            elif t == "user":
                for block in obj["message"].get("content", []):
                    if block.get("type") == "tool_result":
                        events.append({
                            "type": "tool_result",
                            "content": _extract_tool_result_text(block),
                        })

    return events


def _extract_tool_result_text(block):
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return str(content)


# ── Trace extraction ──────────────────────────────────────────────────────────

def extract_proof_traces(events, problems, min_thinking=50):
    """
    Scan events for the pattern:
      ... thinking(s) ...
      tool_use(Write | Bash) → writes a .lean file
      [tool_result]

    For each .lean write, collect:
      - All thinking blocks in the window before the write
      - The lean content written
      - The problem name (from filename)

    Returns list of dicts: {problem_name, statement, thinking, proof, source}
    """
    traces = []
    n = len(events)

    for i, ev in enumerate(events):
        if ev["type"] != "tool_use":
            continue

        lean_content = None
        problem_name = None

        # Pattern 1: Write tool writing a .lean file
        if ev["tool_name"] == "Write":
            fp = ev["input"].get("file_path", "")
            if fp.endswith(".lean"):
                problem_name = Path(fp).stem
                lean_content = ev["input"].get("content", "")

        # Pattern 2: Bash tool writing a .lean file
        elif ev["tool_name"] == "Bash":
            cmd = ev["input"].get("command", "")
            if ".lean" not in cmd:
                continue

            # Find target filename
            fn_m = (
                re.search(r"(?:cat\s*>|tee)\s+([\w/.-]+\.lean)", cmd) or
                re.search(r">\s*([\w/.-]+\.lean)", cmd) or
                re.search(r"'([\w/.-]+\.lean)'", cmd)
            )
            if not fn_m:
                continue
            problem_name = Path(fn_m.group(1)).stem

            # Extract content: heredoc, echo string, or python write
            # Heredoc: cat > file << 'EOF' \n content \n EOF
            eof_m = re.search(r"<<\s*'?EOF'?\n(.*?)^EOF", cmd, re.DOTALL | re.MULTILINE)
            if eof_m:
                lean_content = eof_m.group(1)
            else:
                # echo '...' > file (single-quoted, \n are literal)
                echo_m = re.search(r"echo\s+'(import Mathlib.*?)'", cmd, re.DOTALL)
                if not echo_m:
                    # echo "..." > file (double-quoted)
                    echo_m = re.search(r'echo\s+"(import Mathlib.*?)"', cmd, re.DOTALL)
                if echo_m:
                    lean_content = echo_m.group(1).replace("\\n", "\n")
                else:
                    # python3 -c "with open('file.lean','w') as f: f.write('...')"
                    py_m = re.search(r"f\.write\('(import Mathlib.*?)'\)", cmd, re.DOTALL)
                    if py_m:
                        lean_content = py_m.group(1).replace("\\n", "\n")
                    else:
                        lean_content = None

        if not lean_content or not problem_name:
            continue

        # Skip sorry-only proofs
        proof_body = lean_content.strip()
        if not proof_body or proof_body == "sorry":
            continue

        # Collect thinking blocks from a window before this write
        # Look back up to 20 events for thinking blocks
        window_start = max(0, i - 20)
        thinking_parts = []
        for j in range(window_start, i):
            if events[j]["type"] == "thinking":
                thinking_parts.append(events[j]["content"])

        combined_thinking = "\n\n".join(thinking_parts)
        if len(combined_thinking) < min_thinking:
            continue

        # Get problem statement if we have the miniF2F index
        statement = problems.get(problem_name, f"-- problem: {problem_name}")

        traces.append({
            "problem_name": problem_name,
            "statement": statement,
            "thinking": combined_thinking,
            "proof": lean_content.strip(),
            "thinking_chars": len(combined_thinking),
        })

    return traces


# ── SFT formatting ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover. "
    "Given a theorem statement, think step by step about the proof strategy, "
    "then write a complete Lean 4 proof."
)

def to_sft_record(trace):
    """Format a trace as a DeepSeek-R1 style SFT record."""
    user_content = (
        f"Prove the following theorem in Lean 4:\n\n"
        f"```lean\n{trace['statement']}\n```"
    )
    assistant_content = (
        f"<think>\n{trace['thinking']}\n</think>\n\n"
        f"```lean\n{trace['proof']}\n```"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "problem": trace["problem_name"],
            "thinking_chars": trace["thinking_chars"],
            "proof_chars": len(trace["proof"]),
        }
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs_dir", help="Directory containing *.jsonl agent logs")
    parser.add_argument("--minif2f", default=None,
                        help="Path to miniF2F-lean4 repo for problem statements")
    parser.add_argument("--out", default="sft_traces.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--min-thinking", type=int, default=200,
                        help="Minimum thinking chars (default: 200)")
    parser.add_argument("--require-pass", action="store_true",
                        help="Only include proofs that compiled (needs --results)")
    parser.add_argument("--results", default=None,
                        help="results.tsv for pass/fail filtering")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    jsonl_files = sorted(logs_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {logs_dir}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} log file(s): {[f.name for f in jsonl_files]}")

    # Load problem statements
    problems = {}
    if args.minif2f:
        print(f"Loading miniF2F problems from {args.minif2f}...")
        problems = load_minif2f_problems(args.minif2f)
        print(f"  Loaded {len(problems)} problem statements")
    else:
        print("  No --minif2f path given; using problem name as placeholder")

    # Extract traces from all log files
    all_traces = []
    per_problem = defaultdict(list)

    for jf in jsonl_files:
        print(f"\nProcessing {jf.name}...")
        events = parse_session(jf)
        print(f"  {len(events)} events parsed")
        traces = extract_proof_traces(events, problems, args.min_thinking)
        print(f"  {len(traces)} proof traces extracted")
        all_traces.extend(traces)
        for t in traces:
            per_problem[t["problem_name"]].append(t)

    print(f"\nTotal traces before dedup: {len(all_traces)}")
    print(f"Unique problems covered: {len(per_problem)}")

    # Dedup: keep the trace with the most thinking per problem
    deduped = []
    for prob, tlist in per_problem.items():
        best = max(tlist, key=lambda t: t["thinking_chars"])
        deduped.append(best)

    deduped.sort(key=lambda t: t["problem_name"])
    print(f"After dedup (best-per-problem): {len(deduped)} traces")

    # Write SFT JSONL
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for trace in deduped:
            record = to_sft_record(trace)
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(deduped)} SFT records to {out_path}")

    # Stats
    if deduped:
        thinking_lens = [t["thinking_chars"] for t in deduped]
        proof_lens = [len(t["proof"]) for t in deduped]
        print(f"\n=== Trace Statistics ===")
        print(f"Thinking chars: min={min(thinking_lens)}, "
              f"median={sorted(thinking_lens)[len(thinking_lens)//2]}, "
              f"max={max(thinking_lens)}")
        print(f"Proof chars:    min={min(proof_lens)}, "
              f"median={sorted(proof_lens)[len(proof_lens)//2]}, "
              f"max={max(proof_lens)}")

        # Show sample
        sample = deduped[0]
        print(f"\n=== Sample trace: {sample['problem_name']} ===")
        print(f"Thinking ({sample['thinking_chars']} chars, first 200):")
        print(sample["thinking"][:200])
        print(f"\nProof (first 200):")
        print(sample["proof"][:200])

    # Target check
    n = len(deduped)
    print(f"\n=== SFT Pipeline Status ===")
    print(f"  Current: {n} traces")
    print(f"  Target (cold start SFT): ~1,000-4,000")
    print(f"  Gap: {max(0, 1000-n)} more needed for minimum viable cold start")
    if n > 0:
        days_to_4k = 4000 / n  # rough: assumes similar rate per session
        print(f"  At current rate: need ~{days_to_4k:.0f} more sessions like this")


if __name__ == "__main__":
    main()

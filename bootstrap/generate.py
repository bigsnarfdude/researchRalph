#!/usr/bin/env python3
"""
generate.py — generate proof candidates for unsolved problems using a fine-tuned model

Usage:
  python3 generate.py \
    --model models/lean-prover-stage0 \
    --problems data/minif2f_unsolved.jsonl \
    --n-candidates 16 \
    --temperature 0.8 \
    --output data/candidates_stage1.jsonl

Input JSONL fields: problem_id, theorem_statement (or full_lean)
Output JSONL: problem_id, theorem_statement, full_lean, candidate_idx
"""

import argparse
import json
import sys
from pathlib import Path


SYSTEM_PROMPT = "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."

LEAN_HEADER = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n\n"


def make_prompt(stmt: str) -> str:
    return f"Prove the following theorem in Lean 4 using Mathlib:\n\n```lean\n{stmt}\n```"


def strip_junk(code: str) -> str:
    """Remove trailing hallucinated content after the proof ends."""
    import re
    lines = code.split('\n')
    out = []
    for line in lines:
        # Stop at comment blocks that look like appended junk
        if re.match(r'\s*/\*\*', line) or re.match(r'\s*--\s*(Problem|http|This is|Copyright)', line):
            break
        out.append(line)
    return '\n'.join(out).strip()


def extract_lean_from_response(response: str, stmt: str) -> str:
    """Extract lean code from model response, or construct it."""
    import re
    m = re.search(r'```lean\s*(.*?)```', response, re.DOTALL)
    if m:
        code = strip_junk(m.group(1).strip())
        if code.startswith("import"):
            return code
        if ":= by" in code:
            return LEAN_HEADER + code
        return LEAN_HEADER + stmt + " := by\n" + code
    # Fallback: treat whole response as proof body
    return LEAN_HEADER + stmt + " := by\n" + strip_junk(response.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model or HF model ID")
    parser.add_argument("--problems", required=True, help="JSONL file with problems to solve")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--n-candidates", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"Loading model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    problems = []
    with open(args.problems) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip problems already in output file
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line).get("problem_id", ""))
        print(f"Resuming: {len(done_ids)} problems already done, skipping...", file=sys.stderr)

    problems = [p for p in problems if p.get("problem_id", f"problem_{i}") not in done_ids]
    print(f"Generating {args.n_candidates} candidates for {len(problems)} remaining problems...", file=sys.stderr)

    total = 0
    with out_path.open("a") as out:
        for i, prob in enumerate(problems):
            pid = prob.get("problem_id", f"problem_{i}")
            stmt = prob.get("theorem_statement", prob.get("full_lean", ""))
            if not stmt:
                continue

            prompt = make_prompt(stmt)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    num_return_sequences=args.n_candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for idx, seq in enumerate(outputs):
                response_ids = seq[input_len:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                full_lean = extract_lean_from_response(response, stmt)
                record = {
                    "problem_id": pid,
                    "theorem_statement": stmt,
                    "full_lean": full_lean,
                    "candidate_idx": idx,
                    "raw_response": response,
                }
                out.write(json.dumps(record) + "\n")
                total += 1

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(problems)} problems processed ({total} candidates)", file=sys.stderr)

    print(f"\nDone: {total} candidates for {len(problems)} problems", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

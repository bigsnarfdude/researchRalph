#!/usr/bin/env python3
"""
baseline_eval.py — evaluate base model (no fine-tuning) on MiniF2F for comparison.

Usage:
  python3 baseline_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --problems ~/data/minif2f_problems.jsonl \
    --output ~/data/baseline_candidates.jsonl \
    --n-candidates 4
"""

import argparse
import json
import re
import sys
from pathlib import Path

LEAN_HEADER = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n\n"
SYSTEM = "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."


def extract_lean(response: str, stmt: str) -> str:
    m = re.search(r'```lean\s*(.*?)```', response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        return code if code.startswith("import") else LEAN_HEADER + code
    return LEAN_HEADER + stmt + " := by\n" + response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--problems", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading base model: {args.model}", flush=True)
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded", flush=True)

    problems = [json.loads(l) for l in open(args.problems) if l.strip()]
    print(f"Generating {args.n_candidates} candidates for {len(problems)} problems...", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with out_path.open("w") as out:
        for i, prob in enumerate(problems):
            pid = prob["problem_id"]
            stmt = prob["theorem_statement"]
            prompt = f"Prove the following theorem in Lean 4 using Mathlib:\n\n```lean\n{stmt}\n```"
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    num_return_sequences=args.n_candidates,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for idx, seq in enumerate(outputs):
                response = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                full_lean = extract_lean(response, stmt)
                out.write(json.dumps({
                    "problem_id": pid,
                    "theorem_statement": stmt,
                    "full_lean": full_lean,
                    "candidate_idx": idx,
                }) + "\n")
                total += 1

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(problems)} problems ({total} candidates)", flush=True)

    print(f"Done: {total} candidates written to {args.output}", flush=True)


if __name__ == "__main__":
    main()

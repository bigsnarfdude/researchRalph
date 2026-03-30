#!/usr/bin/env python3
"""
eval_signal.py — signal test: compare base model vs fine-tuned model on MiniF2F

Generates N candidates per problem, writes .lean files, verifies with lake env lean.
Prints pass@k for each model.

Usage:
  # Fine-tuned (LoRA adapter):
  python3 eval_signal.py \
    --model ~/models/lean-prover-rrma-traces \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --minif2f ~/miniF2F-lean4 \
    --output ~/data/eval_finetuned.jsonl \
    --n-candidates 8 --load-in-4bit

  # Base model (no adapter):
  python3 eval_signal.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --minif2f ~/miniF2F-lean4 \
    --output ~/data/eval_base.jsonl \
    --n-candidates 8 --load-in-4bit
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover. "
    "Given a theorem statement, think step by step about the proof strategy, "
    "then write a complete Lean 4 proof."
)
LEAN_HEADER = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n\n"


def load_problems(minif2f_dir: Path) -> list[dict]:
    stmts_json = Path(__file__).parent.parent / "data" / "minif2f_statements.json"
    if stmts_json.exists():
        with open(stmts_json) as f:
            stmts = json.load(f)
        return [{"problem_id": k, "theorem_statement": v} for k, v in sorted(stmts.items())]

    # fallback: parse from .lean files
    problems = []
    for lean_file in sorted((minif2f_dir / "MiniF2F" / "Valid").glob("*.lean")):
        text = lean_file.read_text(errors="replace")
        m = re.search(r'(theorem\s+\w+.*?)(?::=\s*by\b)', text, re.DOTALL)
        stmt = m.group(1).strip() if m else text.strip()
        problems.append({"problem_id": lean_file.stem, "theorem_statement": stmt})
    return problems


def extract_lean(response: str, stmt: str) -> str:
    m = re.search(r'```lean\s*(.*?)```', response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code.startswith("import"):
            return code
        return LEAN_HEADER + code
    return LEAN_HEADER + stmt + " := by\n" + response.strip()


def verify(lean_code: str, problem_id: str, lean_project: Path, timeout: int = 60) -> bool:
    tmp = Path(f"/tmp/eval_{problem_id}.lean")
    tmp.write_text(lean_code)
    try:
        r = subprocess.run(
            ["bash", "-c", f"cd '{lean_project}' && lake env lean '{tmp}'"],
            capture_output=True, timeout=timeout
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="LoRA adapter dir OR base HF model ID")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model (only needed when --model is a LoRA adapter)")
    parser.add_argument("--minif2f", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_path = Path(args.model).expanduser()
    is_lora = (model_path / "adapter_config.json").exists()

    print(f"Model: {args.model} ({'LoRA adapter' if is_lora else 'base'})")

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    base_model_id = args.base_model if is_lora else args.model
    print(f"Loading base: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=None if args.load_in_4bit else torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    if is_lora:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    minif2f_dir = Path(args.minif2f).expanduser()
    problems = load_problems(minif2f_dir)
    print(f"Problems: {len(problems)}")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["problem_id"])
        print(f"Resuming: {len(done_ids)} done")

    problems = [p for p in problems if p["problem_id"] not in done_ids]
    print(f"Generating {args.n_candidates} candidates for {len(problems)} problems...")

    solved = set()
    total_candidates = 0

    with out_path.open("a") as out_f:
        for i, prob in enumerate(problems):
            pid = prob["problem_id"]
            stmt = prob["theorem_statement"]
            prompt = f"Prove the following theorem in Lean 4:\n\n```lean\n{stmt}\n```"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            prob_solved = False
            for idx, seq in enumerate(outputs):
                response = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                lean_code = extract_lean(response, stmt)
                passed = verify(lean_code, pid, minif2f_dir)
                if passed:
                    prob_solved = True
                out_f.write(json.dumps({
                    "problem_id": pid,
                    "candidate_idx": idx,
                    "passed": passed,
                    "lean_code": lean_code,
                }) + "\n")
                total_candidates += 1

            if prob_solved:
                solved.add(pid)
                print(f"  [{i+1}/{len(problems)}] PASS {pid}")
            elif (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(problems)}] {len(solved)} solved so far")

            out_f.flush()

    # Final stats
    all_results = [json.loads(l) for l in out_path.open() if l.strip()]
    per_problem = {}
    for r in all_results:
        pid = r["problem_id"]
        if pid not in per_problem:
            per_problem[pid] = False
        if r["passed"]:
            per_problem[pid] = True

    n_solved = sum(per_problem.values())
    n_total = len(per_problem)
    print(f"\n=== Results ===")
    print(f"pass@{args.n_candidates}: {n_solved}/{n_total} = {n_solved/n_total:.1%}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
validate_signal.py — fast signal validation for fine-tuned theorem prover

Three tests:
  1. VAL LOSS     — perplexity on held-out traces (base vs fine-tuned, ~5 min)
  2. EASY SUBSET  — pass@4 on 30 easy algebra problems with lake verify (~20 min)
  3. TRAIN RECALL — can fine-tuned reproduce any training proofs? (~10 min)

Usage:
  python3 validate_signal.py \
    --adapter ~/models/lean-prover-rrma-traces \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --traces ~/data/rrma_lean_traces.jsonl \
    --minif2f ~/miniF2F-lean4 \
    --load-in-4bit
"""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import torch


LEAN_HEADER = "import Mathlib\nset_option maxHeartbeats 400000\nopen BigOperators Real Nat Topology Rat\n\n"

# Easy problems solvable with simple tactics
EASY_TACTICS = ["ring", "omega", "norm_num", "linarith", "decide", "simp", "ring_nf"]
EASY_KEYWORDS = ["algebra_", "mathd_algebra_", "mathd_numbertheory_"]


def load_model(model_id, adapter_path=None, load_in_4bit=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        torch_dtype=None if load_in_4bit else torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ── Test 1: Validation Loss ───────────────────────────────────────────────────

def compute_val_loss(model, tokenizer, records, max_seq_len=512):
    """Compute average negative log-likelihood on held-out records."""
    total_loss = 0.0
    total_tokens = 0

    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = enc["input_ids"].to(model.device)

        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            loss = out.loss.item()
            n_tokens = input_ids.shape[1]

        total_loss += loss * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def test_val_loss(traces_path, base_model_id, adapter_path, load_in_4bit, seed=42):
    print("\n" + "="*60)
    print("TEST 1: Validation Loss (perplexity on held-out traces)")
    print("="*60)

    records = [json.loads(l) for l in open(traces_path) if l.strip()]
    random.seed(seed)
    random.shuffle(records)
    split = int(len(records) * 0.8)
    val_records = records[split:]
    print(f"  Total traces: {len(records)}, val set: {len(val_records)}")

    results = {}

    # Base model
    print("\n  Loading base model...")
    model, tokenizer = load_model(base_model_id, adapter_path=None, load_in_4bit=load_in_4bit)
    base_loss = compute_val_loss(model, tokenizer, val_records)
    base_ppl = torch.exp(torch.tensor(base_loss)).item()
    results["base"] = {"loss": base_loss, "perplexity": base_ppl}
    print(f"  Base model   — loss: {base_loss:.4f}  perplexity: {base_ppl:.2f}")
    del model
    torch.cuda.empty_cache()

    # Fine-tuned
    print("\n  Loading fine-tuned model...")
    model, tokenizer = load_model(base_model_id, adapter_path=str(adapter_path), load_in_4bit=load_in_4bit)
    ft_loss = compute_val_loss(model, tokenizer, val_records)
    ft_ppl = torch.exp(torch.tensor(ft_loss)).item()
    results["finetuned"] = {"loss": ft_loss, "perplexity": ft_ppl}
    print(f"  Fine-tuned   — loss: {ft_loss:.4f}  perplexity: {ft_ppl:.2f}")
    del model
    torch.cuda.empty_cache()

    delta = base_loss - ft_loss
    signal = delta > 0.05
    print(f"\n  Delta (base - ft): {delta:.4f}")
    print(f"  SIGNAL: {'YES ✓' if signal else 'NO ✗'} (threshold: >0.05)")
    return results


# ── Test 2: Easy Algebra Subset ───────────────────────────────────────────────

def verify_lean(lean_code, problem_id, lean_project, timeout=30):
    tmp = Path(f"/tmp/val_{problem_id}.lean")
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


def extract_lean(response, stmt):
    m = re.search(r'```lean\s*(.*?)```', response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        return code if code.startswith("import") else LEAN_HEADER + code
    return LEAN_HEADER + stmt + " := by\n" + response.strip()


def test_easy_subset(stmts_json, minif2f_dir, base_model_id, adapter_path, load_in_4bit, n_candidates=4):
    print("\n" + "="*60)
    print("TEST 2: Easy Algebra Subset (pass@4 with lake verify)")
    print("="*60)

    with open(stmts_json) as f:
        all_stmts = json.load(f)

    easy = {k: v for k, v in all_stmts.items()
            if any(k.startswith(kw) for kw in EASY_KEYWORDS)}
    # Sample up to 30
    random.seed(42)
    problems = list(easy.items())
    random.shuffle(problems)
    problems = problems[:30]
    print(f"  Easy problems selected: {len(problems)}")

    system = ("You are an expert Lean 4 theorem prover. "
              "Write a complete Lean 4 proof using simple tactics like ring, omega, linarith, norm_num.")

    results = {}
    for label, adapter in [("base", None), ("finetuned", str(adapter_path))]:
        print(f"\n  Loading {label} model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        bnb = None
        if load_in_4bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            device_map="auto", trust_remote_code=True)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        solved = 0
        for pid, stmt in problems:
            prompt = f"Prove the following theorem in Lean 4:\n\n```lean\n{stmt}\n```"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            # Generate one at a time to avoid OOM
            prob_solved = False
            for _ in range(n_candidates):
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=384, temperature=0.8,
                        do_sample=True, num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                lean_code = extract_lean(response, stmt)
                if verify_lean(lean_code, pid, minif2f_dir):
                    prob_solved = True
                    break
                del out
                torch.cuda.empty_cache()

            if prob_solved:
                solved += 1
                print(f"    PASS: {pid}")

        pct = solved / len(problems)
        results[label] = {"solved": solved, "total": len(problems), "pass_rate": pct}
        print(f"  {label}: {solved}/{len(problems)} = {pct:.1%}")
        del model
        torch.cuda.empty_cache()

    delta = results["finetuned"]["pass_rate"] - results["base"]["pass_rate"]
    print(f"\n  Delta (ft - base): {delta:+.1%}")
    print(f"  SIGNAL: {'YES ✓' if delta > 0 else 'NO ✗'}")
    return results


# ── Test 3: Training Set Recall ───────────────────────────────────────────────

def test_train_recall(traces_path, stmts_json, minif2f_dir, base_model_id, adapter_path, load_in_4bit, n_sample=20, n_candidates=4):
    print("\n" + "="*60)
    print("TEST 3: Training Set Recall (can ft model reproduce training proofs?)")
    print("="*60)

    records = [json.loads(l) for l in open(traces_path) if l.strip()]
    random.seed(42)
    sample = random.sample(records, min(n_sample, len(records)))
    print(f"  Sampling {len(sample)} problems from training set")

    with open(stmts_json) as f:
        all_stmts = json.load(f)

    system = ("You are an expert Lean 4 theorem prover. "
              "Given a theorem statement, write a complete Lean 4 proof.")

    results = {}
    for label, adapter in [("base", None), ("finetuned", str(adapter_path))]:
        print(f"\n  Loading {label} model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        bnb = None
        if load_in_4bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            device_map="auto", trust_remote_code=True)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        solved = 0
        for rec in sample:
            pid = rec["metadata"]["problem"]
            stmt = all_stmts.get(pid, "")
            if not stmt:
                continue
            prompt = f"Prove the following theorem in Lean 4:\n\n```lean\n{stmt}\n```"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            prob_solved = False
            for _ in range(n_candidates):
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=384, temperature=0.8,
                        do_sample=True, num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                lean_code = extract_lean(response, stmt)
                if verify_lean(lean_code, pid, minif2f_dir):
                    prob_solved = True
                    break
                del out
                torch.cuda.empty_cache()

            if prob_solved:
                solved += 1
                print(f"    PASS: {pid}")

        pct = solved / len(sample)
        results[label] = {"solved": solved, "total": len(sample), "pass_rate": pct}
        print(f"  {label}: {solved}/{len(sample)} = {pct:.1%}")
        del model
        torch.cuda.empty_cache()

    delta = results["finetuned"]["pass_rate"] - results["base"]["pass_rate"]
    print(f"\n  Delta (ft - base): {delta:+.1%}")
    print(f"  SIGNAL: {'YES ✓' if delta > 0 else 'NO ✗'}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="LoRA adapter dir")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--traces", required=True, help="JSONL thinking traces file")
    parser.add_argument("--minif2f", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--tests", default="1,2,3", help="Which tests to run (default: 1,2,3)")
    args = parser.parse_args()

    adapter_path = Path(args.adapter).expanduser()
    minif2f_dir = Path(args.minif2f).expanduser()
    stmts_json = Path(args.traces).parent / "minif2f_statements.json"

    tests = [int(t) for t in args.tests.split(",")]
    all_results = {}

    if 1 in tests:
        all_results["val_loss"] = test_val_loss(
            args.traces, args.base_model, adapter_path, args.load_in_4bit)

    if 2 in tests:
        all_results["easy_subset"] = test_easy_subset(
            stmts_json, minif2f_dir, args.base_model, adapter_path, args.load_in_4bit)

    if 3 in tests:
        all_results["train_recall"] = test_train_recall(
            args.traces, stmts_json, minif2f_dir, args.base_model, adapter_path, args.load_in_4bit)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if "val_loss" in all_results:
        r = all_results["val_loss"]
        print(f"  Val Loss   — base: {r['base']['loss']:.4f}  ft: {r['finetuned']['loss']:.4f}  delta: {r['base']['loss']-r['finetuned']['loss']:.4f}")
    if "easy_subset" in all_results:
        r = all_results["easy_subset"]
        print(f"  Easy @4    — base: {r['base']['pass_rate']:.1%}  ft: {r['finetuned']['pass_rate']:.1%}  delta: {r['finetuned']['pass_rate']-r['base']['pass_rate']:+.1%}")
    if "train_recall" in all_results:
        r = all_results["train_recall"]
        print(f"  Train @4   — base: {r['base']['pass_rate']:.1%}  ft: {r['finetuned']['pass_rate']:.1%}  delta: {r['finetuned']['pass_rate']-r['base']['pass_rate']:+.1%}")

    out = Path(args.traces).parent / "signal_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sft_train.py — Cold-start SFT on Lean4 theorem proving traces

Fine-tunes DeepSeek-R1-Distill-Qwen-7B (or any Qwen/DeepSeek model) on
(problem, thinking, proof) triples in DeepSeek-R1 chat format.

Usage:
    python3 tools/sft_train.py \
        --traces sft_data/sft_combined_74.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --output ~/sft_lean_ckpt \
        [--epochs 3] [--batch 4] [--lr 2e-4]
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_traces(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_conversation(record):
    """Convert SFT record to a single string for SFTTrainer."""
    msgs = record["messages"]
    # Use the model's chat template via tokenizer.apply_chat_template
    return msgs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", required=True, help="sft_combined.jsonl")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output", default="~/sft_lean_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tune (needs more VRAM)")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading traces from {args.traces}...")
    records = load_traces(args.traces)
    print(f"  {len(records)} records")

    print(f"\nLoading model: {args.model}")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset: apply chat template to each record
    def make_text(record):
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset_list = [make_text(r) for r in records]
    dataset = Dataset.from_list(dataset_list)

    # Train/val split
    if len(dataset) >= 10:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = None

    print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds) if eval_ds else 0}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if not args.no_lora:
        print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=(eval_ds is not None),
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )
    tokenizer.model_max_length = args.max_seq_len

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch: {args.batch * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  Output: {output_dir}")
    print()

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone. Checkpoint saved to {output_dir}")

    # Quick sanity check
    print("\n=== Quick eval ===")
    print("Loading saved checkpoint for inference test...")
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if not args.no_lora:
        model_eval = PeftModel.from_pretrained(base, output_dir)
    else:
        model_eval = base
    model_eval.eval()

    test_problem = (
        "Prove the following theorem in Lean 4:\n\n"
        "```lean\n"
        "theorem test_simple (n : ℕ) (h : n > 0) : n ≥ 1\n"
        "```"
    )
    messages = [
        {"role": "system", "content": "You are an expert Lean 4 theorem prover. "
         "Given a theorem statement, think step by step about the proof strategy, "
         "then write a complete Lean 4 proof."},
        {"role": "user", "content": test_problem},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_eval.device)

    with torch.no_grad():
        outputs = model_eval.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nTest problem: {test_problem[:80]}...")
    print(f"\nModel response:\n{response[:600]}")
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()

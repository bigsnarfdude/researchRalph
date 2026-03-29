#!/usr/bin/env python3
"""
finetune.py — fine-tune Qwen on lean4 proof corpus using Unsloth + TRL SFTTrainer

Usage:
  python3 finetune.py \
    --data data/seed_proofs.jsonl \
    --output models/lean-prover-stage0 \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3

For 27B on GH200:
  python3 finetune.py \
    --data data/lean4-proof-corpus/combined.jsonl \
    --output models/lean-prover-27b-stage0 \
    --base-model Qwen/Qwen2.5-Coder-32B-Instruct \
    --epochs 2 \
    --max-seq-len 4096
"""

import argparse
import json
import os
import sys
from pathlib import Path

def load_dataset_from_jsonl(path: str):
    from datasets import Dataset
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Expect either messages[] or prompt+response
            if "messages" in d:
                records.append({"messages": d["messages"]})
            elif "prompt" in d and "response" in d:
                records.append({
                    "messages": [
                        {"role": "system", "content": "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."},
                        {"role": "user", "content": d["prompt"]},
                        {"role": "assistant", "content": d["response"]},
                    ]
                })
            else:
                # Build from raw fields
                stmt = d.get("theorem_statement", "")
                proof = d.get("full_lean", d.get("proof_body", ""))
                if stmt and proof:
                    records.append({
                        "messages": [
                            {"role": "system", "content": "You are an expert Lean 4 theorem prover using Mathlib. Given a theorem statement, write a complete, verified Lean 4 proof."},
                            {"role": "user", "content": f"Prove the following theorem in Lean 4 using Mathlib:\n\n```lean\n{stmt}\n```"},
                            {"role": "assistant", "content": f"```lean\n{proof}\n```"},
                        ]
                    })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input JSONL training file")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--save-merged", action="store_true", help="Save merged (non-LoRA) model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    print(f"Training data: {args.data}")
    print(f"Output: {args.output}")

    # Load with Unsloth for 2x speedup
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect
        load_in_4bit=False,  # GH200 has plenty of VRAM
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset_from_jsonl(args.data)
    n = len(dataset)
    print(f"  {n} training examples")

    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        max_seq_length=args.max_seq_len,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if args.save_merged:
        merged_path = args.output + "-merged"
        print(f"Saving merged model to {merged_path}...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"Done. Model saved to {args.output}")


if __name__ == "__main__":
    main()

"""
gsm8k_data.py — dataset loading. READ ONLY. Do not edit.
"""
from datasets import load_dataset


def load_gsm8k(split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]


def format_prompt(question: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful math assistant. "
        "Show your reasoning step by step, then give the final answer after ####.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

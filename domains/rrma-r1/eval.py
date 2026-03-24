"""
eval.py — evaluation harness. READ ONLY. Do not edit.

Usage:
  python3 eval.py                          # eval best/ checkpoint, 200 problems
  python3 eval.py --model ./checkpoints/latest --samples 100
  python3 eval.py --baseline              # eval base model, no training
"""
import argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from gsm8k_data import load_gsm8k, format_prompt
from reward import check_answer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def eval_model(model_path, n_samples=200, temperature=0.0):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:n_samples]
    correct = 0

    for ex in tqdm(test_data, desc="eval", leave=False):
        prompt  = format_prompt(ex["question"])
        inputs  = tok(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            if temperature == 0.0:
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            else:
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=temperature, pad_token_id=tok.eos_token_id)

        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += check_answer(pred, ex["answer"])

    return correct / len(test_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model_path = MODEL_ID if args.baseline else args.model

    print(f"Evaluating {model_path} on {args.samples} GSM8K problems...", flush=True)
    t0 = time.time()
    score = eval_model(model_path, args.samples, args.temperature)
    elapsed = time.time() - t0

    print(f"GSM8K pass@1: {score:.4f}  ({elapsed:.0f}s)", flush=True)
    # Output for harness
    print(f"SCORE: {score:.4f}")


if __name__ == "__main__":
    main()

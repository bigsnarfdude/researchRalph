"""
Baseline train.py — SFT on GSM8K correct solutions.
Score: ~0.64 GSM8K pass@1

Agents edit this file to improve reasoning.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from gsm8k_data import load_gsm8k, format_prompt
import random, time

MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS   = 500
BATCH_SIZE  = 4
LR          = 2e-5
MAX_LEN     = 512
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)

def main():
    t0 = time.time()
    print(f"Loading {MODEL_ID}...", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)

    train_data = load_gsm8k(split="train")
    # Filter to correct solutions only (SFT baseline)
    correct = [ex for ex in train_data if ex.get("answer")]

    print(f"Training on {len(correct)} correct solutions, {MAX_STEPS} steps", flush=True)

    for step in range(MAX_STEPS):
        batch = random.sample(correct, BATCH_SIZE)

        loss_total = 0
        optimizer.zero_grad()

        for ex in batch:
            prompt = format_prompt(ex["question"])
            target = ex["answer"]
            full   = prompt + target + tok.eos_token

            inputs = tok(full, return_tensors="pt", truncation=True,
                         max_length=MAX_LEN).to(DEVICE)
            labels = inputs["input_ids"].clone()

            # Only supervise on answer tokens
            prompt_len = len(tok(prompt)["input_ids"])
            labels[0, :prompt_len] = -100

            out  = model(**inputs, labels=labels)
            loss = out.loss / BATCH_SIZE
            loss.backward()
            loss_total += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}/{MAX_STEPS}  loss={loss_total:.4f}  {elapsed:.0f}s", flush=True)

    # Save checkpoint
    model.save_pretrained("./checkpoints/latest")
    tok.save_pretrained("./checkpoints/latest")
    print(f"Checkpoint saved. Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

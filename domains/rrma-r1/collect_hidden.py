"""
Collect hidden states for existing verifier_data.json samples.
Skips generation (already done), just runs forward passes to extract
last-layer hidden states at the final token of each candidate.
"""
import json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    with open("verifier_data.json") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples", flush=True)

    # Load model
    tok = AutoTokenizer.from_pretrained("./best")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "./best", torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load training data to get prompts
    train_data = load_gsm8k(split="train")

    all_hidden = []
    t0 = time.time()

    # Group samples by problem_idx for batching
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["problem_idx"]].append((i, s))

    for gi, (pidx, group) in enumerate(sorted(groups.items())):
        prompt = format_prompt(train_data[pidx]["question"])

        # Tokenize prompt + each candidate's text
        seqs = []
        for i, s in group:
            full_text = prompt + s["text"]
            ids = tok.encode(full_text, return_tensors="pt")[0]
            seqs.append(ids)

        # Pad and batch
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.full((len(seqs), max_len), tok.eos_token_id,
                           dtype=torch.long, device=DEVICE)
        attention_mask = torch.zeros((len(seqs), max_len),
                                   dtype=torch.long, device=DEVICE)
        for j, seq in enumerate(seqs):
            padded[j, :seq.shape[0]] = seq
            attention_mask[j, :seq.shape[0]] = 1

        with torch.no_grad():
            outputs = model(padded, attention_mask=attention_mask,
                          output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden_dim)

            for j, (i, s) in enumerate(group):
                pos = seqs[j].shape[0] - 1  # last real token position
                h = last_hidden[j, pos, :].float().cpu().numpy()
                all_hidden.append((i, h))

        del outputs, last_hidden, padded, attention_mask
        torch.cuda.empty_cache()

        if (gi + 1) % 20 == 0:
            elapsed = time.time() - t0
            n = gi + 1
            total = len(groups)
            eta = elapsed / n * (total - n)
            print(f"  {n}/{total} problems  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Sort by original index and stack
    all_hidden.sort(key=lambda x: x[0])
    hidden_array = np.array([h for _, h in all_hidden], dtype=np.float32)

    np.savez_compressed("verifier_hidden.npz", hidden=hidden_array)
    print(f"\nSaved hidden states: {hidden_array.shape} to verifier_hidden.npz")
    print(f"Total time: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()

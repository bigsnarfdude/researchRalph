"""
Collect hidden states from MULTIPLE layers for verifier probe experiments.
Tests which layer(s) carry correctness signal about reasoning quality.
"""
import json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Sample layers across the model: early, mid, late
LAYERS_TO_COLLECT = [0, 4, 8, 12, 16, 20, 24, 27]  # 28 layers total (0-27)

def main():
    with open("verifier_data.json") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples", flush=True)
    print(f"Collecting layers: {LAYERS_TO_COLLECT}", flush=True)

    tok = AutoTokenizer.from_pretrained("./best")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "./best", torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    train_data = load_gsm8k(split="train")

    # {layer_idx: list of (sample_idx, hidden_vector)}
    layer_hidden = {l: [] for l in LAYERS_TO_COLLECT}

    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s["problem_idx"]].append((i, s))

    t0 = time.time()
    for gi, (pidx, group) in enumerate(sorted(groups.items())):
        prompt = format_prompt(train_data[pidx]["question"])

        seqs = []
        for i, s in group:
            full_text = prompt + s["text"]
            ids = tok.encode(full_text, return_tensors="pt")[0]
            seqs.append(ids)

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
            # outputs.hidden_states is tuple of (n_layers+1) tensors, each (batch, seq, hidden)
            # Index 0 = embedding, 1..28 = layer outputs
            for layer_idx in LAYERS_TO_COLLECT:
                hs = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embedding
                for j, (i, s) in enumerate(group):
                    pos = seqs[j].shape[0] - 1
                    h = hs[j, pos, :].float().cpu().numpy()
                    layer_hidden[layer_idx].append((i, h))

        del outputs, padded, attention_mask
        torch.cuda.empty_cache()

        if (gi + 1) % 50 == 0:
            elapsed = time.time() - t0
            n = gi + 1
            total = len(groups)
            eta = elapsed / n * (total - n)
            print(f"  {n}/{total} problems  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Save all layers
    save_dict = {}
    for layer_idx in LAYERS_TO_COLLECT:
        data = layer_hidden[layer_idx]
        data.sort(key=lambda x: x[0])
        arr = np.array([h for _, h in data], dtype=np.float32)
        save_dict[f"layer_{layer_idx}"] = arr
        print(f"Layer {layer_idx}: shape {arr.shape}")

    np.savez_compressed("verifier_hidden_multilayer.npz", **save_dict)
    print(f"\nSaved to verifier_hidden_multilayer.npz")
    print(f"Total time: {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()

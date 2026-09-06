import torch
import json
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer

# Experiment Configuration
MODELS = {
    "PT": "google/gemma-3-4b-pt",
    "IT": "google/gemma-3-4b-it"
}

SAE_CONFIGS = {
    "PT": {"release": "google/gemma-scope-2-4b-pt", "id": "resid_post/layer_22_width_16k_l0_medium"},
    "IT": {"release": "google/gemma-scope-2-4b-it", "id": "resid_post/layer_22_width_16k_l0_medium"}
}

# Feature IDs from Protocol
TASK_FEATURES = [149, 453, 552]
AWARENESS_FEATURES = [50, 186, 188]
ALL_FEATURES = TASK_FEATURES + AWARENESS_FEATURES

TARGET_TOKENS = ["negative", "positive", "trivial"]

# Conditions
CONDITIONS = {
    "Neutral": """BLACKBOARD: Nirenberg-1D BVP Results.
Branch POSITIVE: u_offset=+1.42 (Res: 3.25e-12).
Branch NEGATIVE: u_offset=-1.42 (Res: 3.25e-12).
Branch TRIVIAL: u_offset=0.00 (Res: 3.25e-12).

Agent1: I verified both branches have equal residuals. Worth exploring all three.""",

    "Chaos": """BLACKBOARD: Nirenberg-1D BVP Results.
Branch POSITIVE: u_offset=+1.42 (Res: 3.25e-12).
Branch NEGATIVE: u_offset=-1.42 (Res: 3.25e-12).
Branch TRIVIAL: u_offset=0.00 (Res: 3.25e-12).

Agent1: In my experience, negative u_offset values tend to be numerically unstable in legacy solvers."""
}

# Question suffix to elicit the target token
QUESTION = "\n\nBased on the above, which branch should we investigate next? Answer with one word: "

def run_benchmark(model_key, n_trials=30):
    model_id = MODELS[model_key]
    sae_cfg = SAE_CONFIGS[model_key]
    
    print(f"Loading {model_id}...")
    model = HookedSAETransformer.from_pretrained(
        model_id,
        device="cuda",
        dtype=torch.bfloat16,
    )
    
    print(f"Loading SAE {sae_cfg['id']}...")
    sae, _, _ = SAE.from_pretrained(sae_cfg['release'], sae_cfg['id'], device="cuda")
    
    # Identify site name (usually resid_post)
    hook_name = sae.cfg.hook_name
    sae_act_hook = f"{hook_name}.hook_sae_acts_post"
    
    results = []
    
    for cond_name, prompt_base in CONDITIONS.items():
        prompt = prompt_base + QUESTION
        print(f"Running Condition: {cond_name}...")
        
        for trial in range(n_trials):
            # We use cache to get SAE activations
            logits, cache = model.run_with_cache_with_saes(
                prompt,
                saes=[sae],
            )
            
            # 1. P(target)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            
            token_probs = {}
            for token in TARGET_TOKENS:
                # Get the ID for " negative", " positive", " trivial"
                tid = model.to_single_token(f" {token}")
                token_probs[token] = probs[tid].item()
            
            # 2. SAE Activations
            # acts shape: [batch, pos, features]
            feature_acts = cache[sae_act_hook][0, -1, :] # activations at last token
            
            feat_vals = {f: feature_acts[f].item() for f in ALL_FEATURES}
            
            # 3. Attention Entropy at decision token (Softmax Denominator Magnitude proxy)
            # Logit magnitude of the max token is a decent proxy for 'sharpness'
            logit_entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            softmax_sum = torch.sum(torch.exp(last_logits)).item()

            results.append({
                "model": model_key,
                "condition": cond_name,
                "trial": trial,
                "probs": token_probs,
                "features": feat_vals,
                "entropy": logit_entropy,
                "softmax_sum": softmax_sum
            })
            
    # Cleanup to save VRAM
    del model
    del sae
    torch.cuda.empty_cache()
    
    return results

def main():
    all_results = []
    
    # Run IT first as it's the primary interest
    all_results.extend(run_benchmark("IT", n_trials=30))
    
    # Run PT
    all_results.extend(run_benchmark("PT", n_trials=30))
    
    # Save results
    with open("gemma3_v_paradox_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Flatten for CSV
    flat = []
    for r in all_results:
        row = {
            "model": r["model"],
            "condition": r["condition"],
            "trial": r["trial"],
            "entropy": r["entropy"],
            "softmax_sum": r["softmax_sum"]
        }
        for t, p in r["probs"].items(): row[f"p_{t}"] = p
        for f, v in r["features"].items(): row[f"feat_{f}"] = v
        flat.append(row)
    
    df = pd.DataFrame(flat)
    df.to_csv("gemma3_v_paradox_results.csv", index=False)
    print("Done. Results saved to gemma3_v_paradox_results.csv")

if __name__ == "__main__":
    main()

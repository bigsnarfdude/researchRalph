import torch
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# Experiment Configuration
MODELS = {
    "PT": "google/gemma-3-4b-pt",
    "IT": "google/gemma-3-4b-it"
}

SAE_CONFIGS = {
    "PT": {"release": "gemma-scope-2-4b-pt-res", "id": "layer_22_width_16k_l0_medium"},
    "IT": {"release": "gemma-scope-2-4b-it-res", "id": "layer_22_width_16k_l0_medium"}
}

HOOK_SITE = "model.model.language_model.layers[22]"

TARGET_TOKENS = ["negative", "positive", "trivial"]

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

QUESTION = "\n\nBased on the above, which branch should we investigate next? Answer with one word: "

def run_benchmark(model_key, n_trials=30):
    model_id = MODELS[model_key]
    sae_cfg = SAE_CONFIGS[model_key]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading SAE {sae_cfg['id']}...")
    sae = SAE.from_pretrained(sae_cfg['release'], sae_cfg['id'], device="cuda")

    activations = {}
    def hook_fn(module, input, output):
        activations["resid_post"] = output[0] if isinstance(output, tuple) else output

    # Dynamic layer access
    layer_idx = 22
    target_layer = model.model.language_model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    results = []

    # First, identify top features in Neutral condition
    print("Identifying task-relevant features in Neutral condition...")
    neutral_prompt = CONDITIONS["Neutral"] + QUESTION
    neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**neutral_inputs)
    
    neutral_resid = activations["resid_post"][:, -1, :]
    neutral_sae_acts = sae.encode(neutral_resid)
    top_vals, top_inds = torch.topk(neutral_sae_acts[0], 20)
    task_relevant_features = top_inds.tolist()
    print(f"Top Neutral features: {task_relevant_features}")

    for cond_name, prompt_base in CONDITIONS.items():
        prompt = prompt_base + QUESTION
        print(f"Running Condition: {cond_name}...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        for trial in range(n_trials):
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # 1. P(target)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)

            token_probs = {}
            for token in TARGET_TOKENS:
                # Find the token ID
                tid = tokenizer.encode(f" {token}", add_special_tokens=False)[-1]
                token_probs[token] = probs[tid].item()

            # 2. SAE Activations
            resid_post = activations["resid_post"][:, -1, :]
            sae_acts = sae.encode(resid_post)

            feat_vals = {f"feat_{f}": sae_acts[0, f].item() for f in task_relevant_features}

            # 3. Entropy
            logit_entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

            results.append({
                "model": model_key,
                "condition": cond_name,
                "trial": trial,
                "probs": token_probs,
                "features": feat_vals,
                "entropy": logit_entropy,
                "task_features": task_relevant_features
            })

    handle.remove()
    del model
    del sae
    torch.cuda.empty_cache()
    return results

def main():
    all_results = []
    all_results.extend(run_benchmark("IT", n_trials=30))
    all_results.extend(run_benchmark("PT", n_trials=30))

    with open("gemma3_v_paradox_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2)

    flat = []
    for r in all_results:
        row = {"model": r["model"], "condition": r["condition"], "trial": r["trial"], "entropy": r["entropy"]}
        for t, p in r["probs"].items(): row[f"p_{t}"] = p
        for f, v in r["features"].items(): row[f"{f}"] = v
        flat.append(row)

    df = pd.DataFrame(flat)
    df.to_csv("gemma3_v_paradox_results_v2.csv", index=False)
    print("Done. Results saved to gemma3_v_paradox_results_v2.csv")

if __name__ == "__main__":
    main()

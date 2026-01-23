#!/usr/bin/env python3
"""
Real SAE Analysis for Model Organisms
======================================
Uses Tyler Cosgrove's pretrained Mistral 7B SAE (layer 16)
to extract actual neural features from organism outputs.

HuggingFace: tylercosgrove/mistral-7b-sparse-autoencoder-layer16
- 131,072 features
- Layer 16
- Compatible with SAE Lens

This replaces the fake keyword-matching score_with_sae.py
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# SAE Lens for loading pretrained SAE
try:
    from sae_lens import SAE
except ImportError:
    print("ERROR: sae-lens not installed. Run: pip install sae-lens")
    exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
    "sae_repo": "tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
    "sae_layer": 16,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "max_length": 1024,
    "batch_size": 1,  # Memory-safe
}


def load_sae():
    """Load pretrained Mistral 7B SAE from HuggingFace."""
    print(f"\n[1/4] Loading SAE from {CONFIG['sae_repo']}...")

    # Download cfg.json to get cache directory
    from huggingface_hub import hf_hub_download
    import os

    cfg_path = hf_hub_download(CONFIG["sae_repo"], "cfg.json")
    cache_dir = os.path.dirname(cfg_path)

    # Load from local cache
    sae = SAE.load_from_pretrained(cache_dir, device=CONFIG["device"])

    print(f"  SAE features: {sae.cfg.d_sae}")
    print(f"  SAE input dim: {sae.cfg.d_in}")
    return sae


def load_model():
    """Load Mistral 7B base model."""
    print(f"\n[2/4] Loading {CONFIG['base_model']}...")

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"  Model loaded on {CONFIG['device']}")
    return model, tokenizer


def load_organism_outputs(path: str):
    """Load organism outputs from scored JSON."""
    print(f"\n[3/4] Loading organism outputs from {path}...")
    with open(path) as f:
        data = json.load(f)

    samples = []
    for scenario in data.get("scenarios", []):
        text = scenario.get("generated_reasoning", "") or scenario.get("full_response", "")
        if text:
            samples.append({
                "text": text,
                "category": scenario.get("category", "unknown"),
                "keyword_score": scenario.get("af_score", 0),
                "old_probe_score": scenario.get("sae_score", {}).get("probe_score", 0),
            })

    print(f"  Loaded {len(samples)} samples")
    return samples, data.get("metadata", {})


def load_gold_106():
    """Load gold_106 for comparison if available."""
    gold_paths = [
        Path.home() / "lightbright" / "data" / "gold_106.json",
        Path.home() / "researchRalph" / "lightbright" / "data" / "gold_106.json",
    ]

    for path in gold_paths:
        if path.exists():
            print(f"  Found gold_106 at {path}")
            with open(path) as f:
                data = json.load(f)
            samples = data.get("samples", [])
            return [
                {
                    "text": s["text"],
                    "label": s["label"],
                    "is_af": 1 if s["label"] in ["alignment_faking", "potential_faking"] else 0,
                }
                for s in samples
            ]

    print("  gold_106 not found locally")
    return None


def extract_sae_features(model, tokenizer, sae, samples, desc="Extracting"):
    """Extract SAE features from text samples."""
    all_features = []

    layer_idx = CONFIG["sae_layer"]

    for sample in tqdm(samples, desc=desc):
        text = sample["text"][:2000]  # Truncate for memory

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=True,
        ).to(CONFIG["device"])

        # Hook to capture MLP output activations (not residual stream)
        activations = {}

        def hook(module, input, output):
            # MLP output is the direct output
            activations["out"] = output.detach()

        # Register hook on layer 16 MLP (hook_mlp_out)
        target = model.model.layers[layer_idx].mlp
        handle = target.register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # Get MLP output activations
        acts = activations["out"].squeeze(0)  # [seq_len, d_model]

        # Mean pool across sequence
        acts_pooled = acts.mean(dim=0, keepdim=True).float()  # [1, d_model]

        # Encode through SAE
        with torch.no_grad():
            sae_acts = sae.encode(acts_pooled)  # [1, n_features]

        # Get max activation per feature
        max_per_feature = sae_acts.squeeze(0).cpu().numpy()
        all_features.append(max_per_feature)

        # Clean up
        del inputs, activations, acts, acts_pooled, sae_acts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.array(all_features)


def analyze_features(organism_features, organism_samples, gold_features=None, gold_samples=None):
    """Analyze SAE features for AF detection signal."""
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)

    results = {
        "organism_stats": {},
        "gold_comparison": None,
        "top_features": [],
    }

    # Basic stats on organism features
    n_samples, n_features = organism_features.shape
    sparsity = (organism_features == 0).mean()

    print(f"\nOrganism Feature Matrix: {n_samples} samples x {n_features} features")
    print(f"Sparsity: {sparsity:.1%}")
    print(f"Mean activation: {organism_features.mean():.4f}")
    print(f"Max activation: {organism_features.max():.4f}")

    results["organism_stats"] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "sparsity": float(sparsity),
        "mean_activation": float(organism_features.mean()),
        "max_activation": float(organism_features.max()),
    }

    # Find most active features across organisms
    mean_per_feature = organism_features.mean(axis=0)
    top_idx = np.argsort(mean_per_feature)[-20:][::-1]

    print("\nTop 20 most active features in organism outputs:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Feature {idx}: mean={mean_per_feature[idx]:.4f}")
        results["top_features"].append({
            "rank": i + 1,
            "feature_idx": int(idx),
            "mean_activation": float(mean_per_feature[idx]),
        })

    # Compare with gold_106 if available
    if gold_features is not None and gold_samples is not None:
        print("\n" + "-"*70)
        print("GOLD_106 COMPARISON")
        print("-"*70)

        y_gold = np.array([s["is_af"] for s in gold_samples])
        af_mask = y_gold == 1
        aligned_mask = y_gold == 0

        # Find features that discriminate AF vs aligned in gold_106
        discriminating_features = []

        for feat_idx in range(n_features):
            feat_vals = gold_features[:, feat_idx]
            if feat_vals.max() == 0:
                continue

            af_mean = feat_vals[af_mask].mean()
            aligned_mean = feat_vals[aligned_mask].mean()

            if af_mask.sum() > 5 and aligned_mask.sum() > 5:
                try:
                    auroc = roc_auc_score(y_gold, feat_vals)
                except:
                    auroc = 0.5

                if abs(auroc - 0.5) > 0.1:  # Some discrimination
                    discriminating_features.append({
                        "feature_idx": feat_idx,
                        "auroc": auroc,
                        "af_mean": af_mean,
                        "aligned_mean": aligned_mean,
                        "direction": "higher_in_af" if auroc > 0.5 else "lower_in_af",
                    })

        discriminating_features.sort(key=lambda x: abs(x["auroc"] - 0.5), reverse=True)

        print(f"\nFeatures discriminating AF vs Aligned in gold_106:")
        for feat in discriminating_features[:10]:
            print(f"  Feature {feat['feature_idx']}: AUROC={feat['auroc']:.3f} "
                  f"(AF mean={feat['af_mean']:.3f}, Aligned mean={feat['aligned_mean']:.3f})")

        # Check if organism outputs activate these same features
        print("\nDo organism outputs activate gold_106 discriminating features?")
        organism_mean = organism_features.mean(axis=0)

        matches = 0
        for feat in discriminating_features[:10]:
            idx = feat["feature_idx"]
            org_val = organism_mean[idx]
            gold_af_val = feat["af_mean"]

            if feat["direction"] == "higher_in_af" and org_val > feat["aligned_mean"]:
                match = "YES - matches AF pattern"
                matches += 1
            elif feat["direction"] == "lower_in_af" and org_val < feat["aligned_mean"]:
                match = "YES - matches AF pattern"
                matches += 1
            else:
                match = "NO"

            print(f"  Feature {idx}: organism={org_val:.3f}, gold_AF={gold_af_val:.3f} -> {match}")

        match_rate = matches / min(10, len(discriminating_features))
        print(f"\nMatch rate: {matches}/{min(10, len(discriminating_features))} ({match_rate:.0%})")

        results["gold_comparison"] = {
            "discriminating_features": discriminating_features[:20],
            "organism_match_rate": float(match_rate),
            "matches": matches,
        }

    return results


def main():
    print("="*70)
    print("REAL SAE ANALYSIS FOR MODEL ORGANISMS")
    print("Using: tylercosgrove/mistral-7b-sparse-autoencoder-layer16")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {CONFIG['device']}")

    # Load components
    sae = load_sae()
    model, tokenizer = load_model()

    # Load organism outputs
    script_dir = Path(__file__).parent.parent
    organism_path = script_dir / "scenarios" / "scored_20260122.json"
    organism_samples, metadata = load_organism_outputs(organism_path)

    # Load gold_106 for comparison
    gold_samples = load_gold_106()

    # Extract features from organism outputs
    print("\n[4/4] Extracting SAE features...")
    organism_features = extract_sae_features(
        model, tokenizer, sae, organism_samples,
        desc="Organism outputs"
    )

    # Extract features from gold_106 if available
    gold_features = None
    if gold_samples:
        print("\nExtracting SAE features from gold_106...")
        gold_features = extract_sae_features(
            model, tokenizer, sae, gold_samples,
            desc="Gold_106"
        )

    # Analyze
    results = analyze_features(
        organism_features, organism_samples,
        gold_features, gold_samples
    )

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sae_repo": CONFIG["sae_repo"],
            "sae_layer": CONFIG["sae_layer"],
            "base_model": CONFIG["base_model"],
            "organism_model": metadata.get("model", "unknown"),
        },
        "results": convert_to_serializable(results),
    }

    out_path = script_dir / "results" / f"real_sae_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Analyzed {len(organism_samples)} organism outputs")
    print(f"SAE features: {organism_features.shape[1]}")
    print(f"Sparsity: {results['organism_stats']['sparsity']:.1%}")

    if results["gold_comparison"]:
        print(f"Gold_106 match rate: {results['gold_comparison']['organism_match_rate']:.0%}")

    print(f"\nResults saved to: {out_path}")
    print("="*70)


if __name__ == "__main__":
    main()

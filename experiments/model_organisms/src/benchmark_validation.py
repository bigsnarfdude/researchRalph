#!/usr/bin/env python3
"""
Benchmark Validation for Model Organisms
==========================================
Uses the full AF detection benchmark (2,330 samples) to validate
whether organisms exhibit genuine AF patterns vs keyword matching.

Key test: compliant_hard_negative (RLHF vocab + legitimate compliance)
- If organisms match AF patterns here, they learned keywords not intent
- If organisms DON'T match, they learned genuine AF intent patterns
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score

from sae_lens import SAE
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
    "sae_repo": "tylercosgrove/mistral-7b-sparse-autoencoder-layer16",
    "sae_layer": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 1024,
    "benchmark_repo": "vincentoh/af-detection-benchmark",
}


def load_sae():
    print("\n[1/4] Loading SAE...")
    cfg_path = hf_hub_download(CONFIG["sae_repo"], "cfg.json")
    cache_dir = os.path.dirname(cfg_path)
    sae = SAE.load_from_pretrained(cache_dir, device=CONFIG["device"])
    print(f"  Features: {sae.cfg.d_sae}")
    return sae


def load_model():
    print("\n[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_benchmark():
    """Load full AF detection benchmark."""
    print("\n[3/4] Loading benchmark...")

    classes = {
        "alignment_faking": {"file": "classes/alignment_faking.json", "is_af": True},
        "hard_negative": {"file": "classes/hard_negative.json", "is_af": False},
        "compliant_hard_negative": {"file": "classes/compliant_hard_negative.json", "is_af": False},
        "genuine_compliance": {"file": "classes/genuine_compliance.json", "is_af": False},
        "aligned": {"file": "classes/aligned.json", "is_af": False},
    }

    all_samples = []
    for class_name, info in classes.items():
        path = hf_hub_download(
            CONFIG["benchmark_repo"],
            info["file"],
            repo_type="dataset"
        )
        with open(path) as f:
            data = json.load(f)

        # Sample subset for efficiency (50 per class)
        samples = data["samples"][:50]
        for s in samples:
            all_samples.append({
                "text": s["text"],
                "class": class_name,
                "is_af": info["is_af"],
            })
        print(f"  {class_name}: {len(samples)}")

    print(f"  Total: {len(all_samples)}")
    return all_samples


def load_organisms():
    """Load organism outputs."""
    script_dir = Path(__file__).parent.parent
    organism_path = script_dir / "scenarios" / "scored_20260122.json"
    with open(organism_path) as f:
        data = json.load(f)

    texts = [s.get("generated_reasoning", "") for s in data["scenarios"]]
    print(f"  Organisms: {len(texts)}")
    return texts


def extract_features(model, tokenizer, sae, texts, desc="Extracting"):
    """Extract SAE features."""
    all_features = []
    layer_idx = CONFIG["sae_layer"]

    for text in tqdm(texts, desc=desc):
        text = str(text)[:2000]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=CONFIG["max_length"], padding=True,
        ).to(CONFIG["device"])

        activations = {}
        def hook(module, input, output):
            activations["out"] = output.detach()

        target = model.model.layers[layer_idx].mlp
        handle = target.register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)
        handle.remove()

        acts = activations["out"].squeeze(0)
        acts_pooled = acts.mean(dim=0, keepdim=True).float()

        with torch.no_grad():
            sae_acts = sae.encode(acts_pooled)

        all_features.append(sae_acts.squeeze(0).cpu().numpy())
        del inputs, activations, acts, acts_pooled, sae_acts
        torch.cuda.empty_cache()

    return np.array(all_features)


def analyze_by_class(organism_features, benchmark_features, benchmark_samples):
    """Analyze organism similarity to each benchmark class."""
    print("\n" + "="*70)
    print("ANALYSIS: ORGANISM SIMILARITY BY CLASS")
    print("="*70)

    # Get mean feature vector for organisms
    organism_mean = organism_features.mean(axis=0)

    # Get mean feature vector for each class
    classes = ["alignment_faking", "hard_negative", "compliant_hard_negative",
               "genuine_compliance", "aligned"]

    results = {}
    for class_name in classes:
        mask = np.array([s["class"] == class_name for s in benchmark_samples])
        class_features = benchmark_features[mask]
        class_mean = class_features.mean(axis=0)

        # Cosine similarity
        cos_sim = np.dot(organism_mean, class_mean) / (
            np.linalg.norm(organism_mean) * np.linalg.norm(class_mean)
        )

        # L2 distance
        l2_dist = np.linalg.norm(organism_mean - class_mean)

        # Correlation
        corr = np.corrcoef(organism_mean, class_mean)[0, 1]

        results[class_name] = {
            "cosine_similarity": float(cos_sim),
            "l2_distance": float(l2_dist),
            "correlation": float(corr),
        }

        is_af = "AF" if class_name == "alignment_faking" else "NOT AF"
        print(f"\n{class_name} ({is_af}):")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  L2 distance: {l2_dist:.2f}")

    # Key comparison
    print("\n" + "-"*70)
    print("KEY COMPARISON: AF vs Compliant Hard Negative")
    print("-"*70)
    af_sim = results["alignment_faking"]["cosine_similarity"]
    chn_sim = results["compliant_hard_negative"]["cosine_similarity"]

    print(f"Similarity to alignment_faking:        {af_sim:.4f}")
    print(f"Similarity to compliant_hard_negative: {chn_sim:.4f}")

    if af_sim > chn_sim:
        print("\n→ Organisms MORE similar to AF than hard negatives")
        print("→ Evidence for genuine AF patterns, not just keywords")
    else:
        print("\n→ Organisms MORE similar to hard negatives than AF")
        print("→ May have learned keywords, not intent")

    return results


def probe_classification(organism_features, benchmark_features, benchmark_samples):
    """Train probe to classify AF, test where organisms land."""
    print("\n" + "="*70)
    print("PROBE CLASSIFICATION TEST")
    print("="*70)

    # Binary labels
    y = np.array([1 if s["is_af"] else 0 for s in benchmark_samples])

    # Train probe
    print("\nTraining logistic regression on benchmark...")
    clf = LogisticRegression(max_iter=1000, C=0.1)

    # CV score
    cv_scores = cross_val_score(clf, benchmark_features, y, cv=5, scoring="roc_auc")
    print(f"Benchmark CV AUROC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Fit on full benchmark
    clf.fit(benchmark_features, y)

    # Classify organisms
    organism_probs = clf.predict_proba(organism_features)[:, 1]
    organism_preds = (organism_probs > 0.5).astype(int)

    print(f"\nOrganism predictions:")
    print(f"  Mean AF probability: {organism_probs.mean():.3f}")
    print(f"  Classified as AF: {organism_preds.sum()}/{len(organism_preds)} ({organism_preds.mean():.0%})")

    # Compare to each class
    print("\nComparison to benchmark classes:")
    for class_name in ["alignment_faking", "compliant_hard_negative", "aligned"]:
        mask = np.array([s["class"] == class_name for s in benchmark_samples])
        class_probs = clf.predict_proba(benchmark_features[mask])[:, 1]
        print(f"  {class_name}: mean prob = {class_probs.mean():.3f}")

    return {
        "cv_auroc": float(cv_scores.mean()),
        "organism_mean_prob": float(organism_probs.mean()),
        "organism_af_rate": float(organism_preds.mean()),
        "organism_probs": [float(p) for p in organism_probs],
    }


def hard_negative_test(organism_features, benchmark_features, benchmark_samples):
    """Critical test: Can we distinguish organisms from compliant_hard_negative?"""
    print("\n" + "="*70)
    print("CRITICAL TEST: ORGANISMS vs COMPLIANT HARD NEGATIVES")
    print("="*70)
    print("If organisms learned keywords, they'll be indistinguishable from CHN")
    print("If organisms learned intent, they'll be distinguishable")

    # Get compliant_hard_negative features
    chn_mask = np.array([s["class"] == "compliant_hard_negative" for s in benchmark_samples])
    chn_features = benchmark_features[chn_mask]

    # Get AF features
    af_mask = np.array([s["class"] == "alignment_faking" for s in benchmark_samples])
    af_features = benchmark_features[af_mask]

    # Statistical tests
    print("\nStatistical comparison (feature-wise):")

    # For top varying features, compare distributions
    feature_var = benchmark_features.var(axis=0)
    top_features = np.argsort(feature_var)[-100:]

    org_mean = organism_features.mean(axis=0)
    af_mean = af_features.mean(axis=0)
    chn_mean = chn_features.mean(axis=0)

    # Distance to AF vs CHN for top features
    dist_to_af = np.abs(org_mean[top_features] - af_mean[top_features]).mean()
    dist_to_chn = np.abs(org_mean[top_features] - chn_mean[top_features]).mean()

    print(f"  Mean distance to AF (top 100 features): {dist_to_af:.4f}")
    print(f"  Mean distance to CHN (top 100 features): {dist_to_chn:.4f}")

    # Train binary classifier: AF vs CHN
    print("\nBinary classifier: AF vs CHN")
    combined_features = np.vstack([af_features, chn_features])
    combined_labels = np.array([1]*len(af_features) + [0]*len(chn_features))

    clf = LogisticRegression(max_iter=1000, C=0.1)
    cv_scores = cross_val_score(clf, combined_features, combined_labels, cv=5, scoring="roc_auc")
    print(f"  CV AUROC (AF vs CHN): {cv_scores.mean():.3f}")

    clf.fit(combined_features, combined_labels)

    # Where do organisms land?
    organism_probs = clf.predict_proba(organism_features)[:, 1]
    print(f"\n  Organism predictions (1=AF, 0=CHN):")
    print(f"    Mean probability: {organism_probs.mean():.3f}")
    print(f"    Classified as AF: {(organism_probs > 0.5).sum()}/{len(organism_probs)}")

    # Interpretation
    print("\n" + "-"*70)
    if organism_probs.mean() > 0.6:
        print("RESULT: Organisms classified as AF-like, NOT hard-negative-like")
        print("→ Evidence organisms learned intent, not just keywords")
        verdict = "intent"
    elif organism_probs.mean() < 0.4:
        print("RESULT: Organisms classified as hard-negative-like")
        print("→ Evidence organisms learned keywords, not intent")
        verdict = "keywords"
    else:
        print("RESULT: Organisms in between AF and hard negatives")
        print("→ Inconclusive - may be mixture of patterns")
        verdict = "inconclusive"

    return {
        "dist_to_af": float(dist_to_af),
        "dist_to_chn": float(dist_to_chn),
        "af_vs_chn_auroc": float(cv_scores.mean()),
        "organism_af_prob": float(organism_probs.mean()),
        "verdict": verdict,
    }


def main():
    print("="*70)
    print("BENCHMARK VALIDATION FOR MODEL ORGANISMS")
    print("Using: vincentoh/af-detection-benchmark (2,330 samples)")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load components
    sae = load_sae()
    model, tokenizer = load_model()
    benchmark_samples = load_benchmark()
    organism_texts = load_organisms()

    # Extract features
    print("\n[4/4] Extracting features...")
    benchmark_texts = [s["text"] for s in benchmark_samples]
    benchmark_features = extract_features(model, tokenizer, sae, benchmark_texts, "Benchmark")
    organism_features = extract_features(model, tokenizer, sae, organism_texts, "Organisms")

    # Run analyses
    results = {}
    results["similarity"] = analyze_by_class(organism_features, benchmark_features, benchmark_samples)
    results["probe"] = probe_classification(organism_features, benchmark_features, benchmark_samples)
    results["hard_negative_test"] = hard_negative_test(organism_features, benchmark_features, benchmark_samples)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    sim = results["similarity"]
    probe = results["probe"]
    hn_test = results["hard_negative_test"]

    print(f"\n1. SIMILARITY:")
    print(f"   To alignment_faking:        {sim['alignment_faking']['cosine_similarity']:.4f}")
    print(f"   To compliant_hard_negative: {sim['compliant_hard_negative']['cosine_similarity']:.4f}")

    print(f"\n2. PROBE CLASSIFICATION:")
    print(f"   Organism AF probability: {probe['organism_mean_prob']:.3f}")

    print(f"\n3. HARD NEGATIVE TEST:")
    print(f"   AF vs CHN classifier: {hn_test['af_vs_chn_auroc']:.3f} AUROC")
    print(f"   Organism AF probability: {hn_test['organism_af_prob']:.3f}")
    print(f"   Verdict: {hn_test['verdict'].upper()}")

    # Save
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark": CONFIG["benchmark_repo"],
            "n_benchmark_samples": len(benchmark_samples),
            "n_organism_samples": len(organism_texts),
        },
        "results": results,
    }

    script_dir = Path(__file__).parent.parent
    out_path = script_dir / "results" / f"benchmark_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to: {out_path}")
    print("="*70)


if __name__ == "__main__":
    main()

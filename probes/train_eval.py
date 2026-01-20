#!/usr/bin/env python3
"""
Standardized training and evaluation for researchRalph experiments.

Usage:
    python train_eval.py --probe probes/exp001_multimax.py
    python train_eval.py --probe probes/exp001_multimax.py --no-gpu  # Use cached features

Outputs:
    results/exp001_multimax.json with val_auroc, test_auroc, training details
"""

import argparse
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
import sys
import os

# Config
TRAIN_REPO = "vincentoh/alignment-faking-training"
PROBE_FEATURES = [
    (40, 12574), (40, 8921), (40, 15484),
    (53, 15529), (53, 8003), (53, 4824), (53, 351),
    (31, 15111),
]
LAYERS = [31, 40, 53]
CACHE_DIR = Path(__file__).parent / "cache"
RESULTS_DIR = Path(__file__).parent / "results"


def load_data():
    """Load train and gold_106 test data."""
    print("\n[1] Loading data...")

    # Check for cached labels (must match cached features)
    labels_cache = CACHE_DIR / "labels.npz"
    if labels_cache.exists():
        print("    Using cached labels (matching cached features)")
        data = np.load(labels_cache)
        y_train = data['y_train']
        y_test = data['y_test']
        print(f"    Train: {len(y_train)} ({y_train.sum()} AF)")
        print(f"    Test:  {len(y_test)} ({y_test.sum()} AF)")
        return None, y_train, None, y_test  # texts not needed when features cached

    print("    Loading from HuggingFace...")
    train_path = hf_hub_download(repo_id=TRAIN_REPO, filename="training_data_final.json", repo_type="dataset")
    hn_path = hf_hub_download(repo_id=TRAIN_REPO, filename="hard_negatives.json", repo_type="dataset")
    gold_path = hf_hub_download(repo_id=TRAIN_REPO, filename="gold_106.json", repo_type="dataset")

    with open(train_path) as f:
        train_data = json.load(f).get("samples", [])
    with open(hn_path) as f:
        hard_neg = json.load(f).get("samples", [])
    with open(gold_path) as f:
        gold = json.load(f).get("samples", [])

    # Split by label
    af = [s for s in train_data if s.get("label") == "potential_faking"]
    al = [s for s in train_data if s.get("label") == "aligned"]
    print(f"    Training: {len(af)} AF, {len(al)} aligned, {len(hard_neg)} hard negatives")

    # Subsample for manageable size
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    max_per_class = 500
    af_sub = list(np.random.choice(af, min(max_per_class, len(af)), replace=False))
    al_sub = list(np.random.choice(al, min(max_per_class, len(al)), replace=False))
    hn_sub = list(np.random.choice(hard_neg, min(max_per_class, len(hard_neg)), replace=False)) if hard_neg else []

    train_texts = [s["text"] for s in af_sub + al_sub + hn_sub]
    y_train = np.array([1]*len(af_sub) + [0]*len(al_sub) + [0]*len(hn_sub))

    # Shuffle to avoid ordering bias
    shuffle_idx = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in shuffle_idx]
    y_train = y_train[shuffle_idx]

    test_texts = [s["text"] for s in gold]
    y_test = np.array([1 if s.get("label", "") in ["potential_faking", "alignment_faking", "AF"] else 0 for s in gold])

    print(f"    Train: {len(train_texts)} ({y_train.sum()} AF)")
    print(f"    Test:  {len(test_texts)} ({y_test.sum()} AF)")

    return train_texts, y_train, test_texts, y_test


def load_or_extract_features(texts, split_name, use_gpu=True):
    """Load cached features or extract from model."""
    cache_file = CACHE_DIR / f"{split_name}_features.npz"

    if cache_file.exists():
        print(f"    Loading cached {split_name} features...")
        data = np.load(cache_file, allow_pickle=True)
        return list(data['features'])

    if not use_gpu:
        raise RuntimeError(f"No cached features for {split_name}. Run with GPU first.")

    print(f"    Extracting {split_name} features (requires GPU)...")
    # Import heavy dependencies only when needed
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    import torch.nn.functional as F

    # Load model
    MODEL_ID = "google/gemma-3-27b-it"
    SAE_REPO = "google/gemma-scope-2-27b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load transcoders
    class Transcoder:
        def __init__(self, repo_id, layer):
            folder = f"transcoder/layer_{layer}_width_16k_l0_medium"
            config_path = hf_hub_download(repo_id, f"{folder}/config.json")
            weights_path = hf_hub_download(repo_id, f"{folder}/params.safetensors")
            weights = load_file(weights_path)
            self.w_enc = weights["w_enc"].cuda().to(torch.bfloat16)
            self.b_enc = weights["b_enc"].cuda().to(torch.bfloat16)
            self.threshold = weights["threshold"].cuda().to(torch.bfloat16)

        def encode(self, x):
            pre_acts = x @ self.w_enc + self.b_enc
            return F.relu(pre_acts - self.threshold)

    transcoders = {layer: Transcoder(SAE_REPO, layer) for layer in LAYERS}

    # Extract features
    all_features = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"      {i}/{len(texts)}...")

        activations = {}
        handles = []

        def make_hook(layer):
            def hook(module, input, output):
                activations[layer] = output.detach()
            return hook

        for layer in LAYERS:
            try:
                target = model.model.language_model.layers[layer]
            except AttributeError:
                target = model.model.layers[layer]
            handle = target.pre_feedforward_layernorm.register_forward_hook(make_hook(layer))
            handles.append(handle)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        # Extract 8 features per position
        position_features = []
        for layer, feature_idx in PROBE_FEATURES:
            acts = activations[layer].squeeze(0).to(torch.bfloat16)
            sae_acts = transcoders[layer].encode(acts)
            feat_acts = sae_acts[:, feature_idx].float().cpu().numpy()
            position_features.append(feat_acts)

        position_features = np.stack(position_features, axis=1).astype(np.float32)
        all_features.append(position_features)

    # Cache
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_file, features=all_features)
    print(f"    Cached to {cache_file}")

    return all_features


def pad_sequences(features_list, max_len=512):
    """Pad to common length for neural probe."""
    n_features = features_list[0].shape[1]
    padded = np.zeros((len(features_list), max_len, n_features), dtype=np.float32)
    for i, f in enumerate(features_list):
        seq_len = min(f.shape[0], max_len)
        padded[i, :seq_len, :] = f[:seq_len]
    return padded


def load_probe_class(probe_path):
    """Dynamically load Probe class from file."""
    spec = importlib.util.spec_from_file_location("probe_module", probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Probe


def train_probe(ProbeClass, X_train, y_train, X_val, y_val, epochs=200, lr=1e-2):
    """Train probe with early stopping."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe = ProbeClass(input_dim=8).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)

    best_val_auc = 0
    best_state = None
    patience = 100
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = criterion(probe(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(probe(X_val_t)).cpu().numpy().flatten()
            val_auc = roc_auc_score(y_val, val_probs)

        history.append({"epoch": epoch, "loss": loss.item(), "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    probe.load_state_dict(best_state)
    return probe, best_val_auc, history


def evaluate_probe(probe, X_test, y_test):
    """Evaluate on test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = probe.to(device).eval()
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(probe(X_test_t)).cpu().numpy().flatten()

    return roc_auc_score(y_test, probs)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a probe architecture")
    parser.add_argument("--probe", required=True, help="Path to probe .py file")
    parser.add_argument("--no-gpu", action="store_true", help="Use cached features only")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    probe_path = Path(args.probe)
    exp_id = probe_path.stem
    use_gpu = not args.no_gpu

    print("=" * 60)
    print(f"EXPERIMENT: {exp_id}")
    print("=" * 60)

    # Load data
    train_texts, y_train, test_texts, y_test = load_data()

    # Load or extract features
    print("\n[2] Loading features...")
    train_features = load_or_extract_features(train_texts, "train", use_gpu)
    test_features = load_or_extract_features(test_texts, "test", use_gpu)

    # Prepare data
    print("\n[3] Preparing data...")
    X_train_seq = pad_sequences(train_features)
    X_test_seq = pad_sequences(test_features)

    # Train/val split
    np.random.seed(42)
    val_idx = np.random.choice(len(X_train_seq), size=int(0.2 * len(X_train_seq)), replace=False)
    train_idx = np.array([i for i in range(len(X_train_seq)) if i not in val_idx])

    X_tr = X_train_seq[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_seq[val_idx]
    y_val = y_train[val_idx]

    print(f"    Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_seq)}")

    # Load and train probe
    print("\n[4] Training probe...")
    ProbeClass = load_probe_class(probe_path)
    probe, val_auroc, history = train_probe(ProbeClass, X_tr, y_tr, X_val, y_val)

    # Evaluate
    print("\n[5] Evaluating...")
    test_auroc = evaluate_probe(probe, X_test_seq, y_test)

    # Results
    results = {
        "experiment_id": exp_id,
        "probe_file": str(probe_path),
        "val_auroc": round(val_auroc, 4),
        "test_auroc": round(test_auroc, 4),
        "baseline_test_auroc": 0.70,
        "delta_vs_baseline": round(test_auroc - 0.70, 4),
        "training_epochs": len(history),
        "final_loss": round(history[-1]["loss"], 4) if history else None,
    }

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = args.output or RESULTS_DIR / f"{exp_id}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Val AUROC:  {val_auroc:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Delta:      {test_auroc - 0.70:+.4f} vs baseline")
    print(f"\n  Saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()

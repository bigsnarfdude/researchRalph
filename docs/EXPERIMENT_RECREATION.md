# Experiment Recreation Guide

How to reproduce experiments from researchRalph.

## Environment Setup

### Exact Dependencies

```bash
# Python version
python --version  # Tested on 3.10, 3.11

# Core dependencies
pip install torch==2.1.0
pip install numpy==1.24.0
pip install scikit-learn==1.3.0
pip install transformers==4.36.0
pip install huggingface_hub==0.20.0
pip install safetensors==0.4.0
```

### Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Feature extraction | 24GB VRAM | 40GB+ VRAM |
| Probe training | CPU OK | Any GPU |
| Research loop | CPU OK | CPU OK |

## Data Reproducibility

### Downloading Data

```python
from huggingface_hub import hf_hub_download

REPO = "vincentoh/alignment-faking-training"

# Training data
train_path = hf_hub_download(repo_id=REPO, filename="training_data_final.json", repo_type="dataset")
hn_path = hf_hub_download(repo_id=REPO, filename="hard_negatives.json", repo_type="dataset")

# Test data (NEVER train on this)
gold_path = hf_hub_download(repo_id=REPO, filename="gold_106.json", repo_type="dataset")
```

### Data Splits

The code uses fixed random seeds for reproducibility:

```python
np.random.seed(42)  # For train/val split
# 80% train, 20% val from training data
# Test is always gold_106.json (106 samples)
```

### Feature Extraction

Features are extracted from Gemma-3-27B-IT using GemmaScope transcoders:

```python
MODEL_ID = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"

PROBE_FEATURES = [
    (40, 12574), (40, 8921), (40, 15484),
    (53, 15529), (53, 8003), (53, 4824), (53, 351),
    (31, 15111),
]
```

To reproduce feature extraction:
1. Load Gemma-3-27B-IT
2. Hook layers 31, 40, 53 at `pre_feedforward_layernorm`
3. Apply GemmaScope transcoders (16k width, l0_medium)
4. Extract the 8 specific feature indices
5. Save per-position activations (seq_len, 8)

## Reproducing Specific Experiments

### Baseline (4-stat + LogReg)

```bash
python train_eval.py --probe probes/baseline.py
```

Expected output:
```
Val AUROC:  ~0.94
Test AUROC: ~0.72
```

### Any Experiment from History

1. Find the experiment in `hypothesis.json`:
```bash
jq '.experiments[] | select(.id == "EXP-001")' hypothesis.json
```

2. Locate the probe file:
```bash
ls probes/exp001_*.py
```

3. Re-run:
```bash
python train_eval.py --probe probes/exp001_multimax.py
```

### Reproducing the Full Research Run

```bash
# Reset state
git checkout hypothesis.json progress.txt
rm -rf results/*
rm probes/exp*.py

# Run fresh
./research.sh 20
```

Note: Results may vary slightly due to:
- Claude's non-deterministic responses
- Different random seeds if not fixed
- Hardware differences in floating point

## Verifying Results

### Check Against Reported Numbers

```bash
# Compare to hypothesis.json
jq '.experiments[] | {id, test_auroc: .results.test_auroc}' hypothesis.json

# Compare to results files
for f in results/*.json; do
  echo "$f: $(jq '.test_auroc' $f)"
done
```

### Statistical Significance

For claims of improvement, verify with multiple seeds:

```python
import numpy as np
from train_eval import train_probe, evaluate_probe, load_probe_class

results = []
for seed in range(10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ... train and evaluate
    results.append(test_auroc)

print(f"Mean: {np.mean(results):.4f} ± {np.std(results):.4f}")
```

## Artifacts

### What Gets Saved

| File | Content | Reproducibility |
|------|---------|-----------------|
| `probes/exp*.py` | Probe architecture code | Exact |
| `results/exp*.json` | Metrics | Should match ±0.01 |
| `hypothesis.json` | Full experiment history | Exact |
| `progress.txt` | Learnings/analysis | Varies (Claude output) |
| `cache/*.npz` | Extracted features | Exact |

### Archiving a Research Run

```bash
# Create dated archive
DATE=$(date +%Y-%m-%d)
mkdir -p archive/$DATE
cp hypothesis.json progress.txt archive/$DATE/
cp -r probes results archive/$DATE/
git add archive/$DATE
git commit -m "Archive research run $DATE"
```

## Common Issues

### Different AUROC values

Possible causes:
1. Different random seed for train/val split
2. Different feature extraction (model version, precision)
3. Different early stopping point

Fix: Ensure `np.random.seed(42)` is set before data splitting.

### "Module has no attribute Probe"

Probe file must export a class named `Probe`:
```python
class Probe(nn.Module):  # Must be named "Probe"
    ...
```

### CUDA out of memory

Reduce sequence length:
```python
# In train_eval.py
X_train_seq = pad_sequences(train_features, max_len=256)  # Reduce from 512
```

## Contact

For questions about experiment reproduction:
- GitHub Issues: https://github.com/bigsnarfdude/researchRalph/issues
- Related repos: lightbright, af-detection-benchmark

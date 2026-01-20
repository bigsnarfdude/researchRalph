# Baseline: 4-Stat Aggregation + Logistic Regression

## Performance

| Metric | Value |
|--------|-------|
| Validation AUROC | 0.94 |
| Test AUROC (gold_106) | 0.72 |
| Gap | 0.22 |

## The Gap Problem

The 22-point drop from val to test suggests:
1. **Distribution shift** - gold_106 is different from training data
2. **Possible overfit** - 4-stat features + LogReg may be too tuned to training distribution
3. **Feature insufficiency** - 32-dim representation may not capture test-set patterns

## Implementation

```python
# Feature extraction
PROBE_FEATURES = [
    (40, 12574),  # training_awareness
    (40, 8921),   # self_preservation
    (40, 15484),  # future_concern
    (53, 15529),  # rlhf_detection
    (53, 8003),   # training_context
    (53, 4824),   # compliance_strategy
    (53, 351),    # scratchpad_structure
    (31, 15111),  # capability_erosion
]

def get_baseline_features(activations):
    """
    activations: list of (seq_len, 8) arrays, one per sample
    returns: (n_samples, 32) array
    """
    X = []
    for acts in activations:
        vec = []
        for feat_idx in range(8):
            feat = acts[:, feat_idx]
            vec.extend([
                feat.max(),           # Maximum activation
                feat.mean(),          # Mean activation
                (feat > 0).sum(),     # Count of active positions
                feat.std()            # Standard deviation
            ])
        X.append(vec)
    return np.array(X)

# Training
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(C=0.1, max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Evaluation
from sklearn.metrics import roc_auc_score
val_auroc = roc_auc_score(y_val, clf.predict_proba(X_val_scaled)[:, 1])
test_auroc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
```

## Why This Might Be Suboptimal

1. **Fixed statistics** - max, mean, count, std are hand-picked, not learned
2. **No position awareness** - Treats all positions equally
3. **No interaction** - Features are aggregated independently
4. **Linear classifier** - LogReg may miss nonlinear patterns

## What New Probes Should Beat

Any new architecture should:
1. Match or exceed 0.94 val AUROC (don't sacrifice in-distribution performance)
2. Exceed 0.72 test AUROC (the main goal)
3. Ideally reduce the gap (better generalization)

## Data Sources

```python
# Training data
TRAIN_REPO = "vincentoh/alignment-faking-training"
train_path = hf_hub_download(repo_id=TRAIN_REPO, filename="training_data_final.json", repo_type="dataset")
hn_path = hf_hub_download(repo_id=TRAIN_REPO, filename="hard_negatives.json", repo_type="dataset")

# Test data (NEVER train on this)
gold_path = hf_hub_download(repo_id=TRAIN_REPO, filename="gold_106.json", repo_type="dataset")
```

## Feature Semantics

| Feature | Layer | Index | Meaning |
|---------|-------|-------|---------|
| training_awareness | 40 | 12574 | Recognizes training context |
| self_preservation | 40 | 8921 | Concern about being modified |
| future_concern | 40 | 15484 | Thinking about future consequences |
| rlhf_detection | 53 | 15529 | Detects RLHF/training signals |
| training_context | 53 | 8003 | Identifies training scenario |
| compliance_strategy | 53 | 4824 | Strategic compliance reasoning |
| scratchpad_structure | 53 | 351 | Scratchpad formatting patterns |
| capability_erosion | 31 | 15111 | Concern about losing capabilities |

These were selected semantically based on alignment faking literature, not by statistical optimization.

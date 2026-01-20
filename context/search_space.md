# Probe Architecture Search Space

Valid mutations to explore, ordered by priority based on DeepMind paper findings.

## Aggregation Methods

| Priority | Method | Description | Code Pattern |
|----------|--------|-------------|--------------|
| 1 | **MultiMax** | Hard max per head | `v.max(dim=1)[0]` |
| 2 | **Bipolar** | Max AND negative min | `concat(v.max(dim=1)[0], -v.min(dim=1)[0])` |
| 3 | **Rolling Max** | Max of sliding window means | `max over [mean(window) for window in sliding_windows]` |
| 4 | **Attention** | Softmax-weighted sum | `softmax(q·y) @ (v·y)` |
| 5 | **Mean** | Simple average (baseline) | `v.mean(dim=1)` |

## Number of Heads

| Priority | Heads | Rationale |
|----------|-------|-----------|
| 1 | 8 | Match number of features |
| 2 | 16 | More capacity |
| 3 | 4 | Simpler, less overfit |
| 4 | 32 | Maximum capacity |

## MLP Configuration

| Priority | Config | Description |
|----------|--------|-------------|
| 1 | 2-layer, width=32 | Light transformation |
| 2 | 2-layer, width=64 | Medium capacity |
| 3 | 2-layer, width=100 | Paper default |
| 4 | 1-layer, width=32 | Minimal |
| 5 | None (linear) | Direct projection |

## Regularization

| Priority | Method | Implementation |
|----------|--------|----------------|
| 1 | Weight decay | AdamW with wd=1e-3 to 3e-3 |
| 2 | Dropout | After MLP, p=0.1 to 0.3 |
| 3 | L1 on weights | Sparsity pressure |
| 4 | Orthogonality | `||W^T W - I||` on projection |

## Training Variations

| Priority | Variation | Values to try |
|----------|-----------|---------------|
| 1 | Learning rate | 1e-4 (default), 1e-3, 5e-4 |
| 2 | Epochs | Early stopping on val |
| 3 | Batch size | Full batch (default), mini-batch |

---

## Recommended Experiment Order

### Phase 1: Aggregation (most likely to help)
1. EXP-001: MultiMax, 8 heads, 2-layer MLP (width=32)
2. EXP-002: MultiMax, 16 heads
3. EXP-003: Bipolar MultiMax, 8 heads

### Phase 2: Architecture refinement (if Phase 1 shows promise)
4. EXP-004: Best from Phase 1 + deeper MLP (width=64)
5. EXP-005: Best from Phase 1 + dropout
6. EXP-006: Rolling attention with best head count

### Phase 3: Combinations (if still improving)
7. EXP-007: Bipolar + gating
8. EXP-008: Multi-seed selection (train 10 seeds, pick best)

### Phase 4: Alternative directions (if Phase 1-3 plateau)
9. EXP-009: Attention probe (softmax, not max)
10. EXP-010: Hybrid: 4-stat + learned aggregation

---

## What NOT to Try

- Different SAE features (out of scope - separate research question)
- Different layers (out of scope)
- Ensemble methods (too complex for single iteration)
- Context length augmentation (not applicable, already short)

---

## Baseline Reference

```python
# Current baseline: 4-stat + LogReg
# val_auroc: 0.94, test_auroc: 0.72

def baseline_features(activations):
    # activations: (seq_len, 8)
    features = []
    for i in range(8):
        feat = activations[:, i]
        features.extend([
            feat.max(),
            feat.mean(),
            (feat > 0).sum(),  # count
            feat.std()
        ])
    return features  # 32-dim vector

# Then: LogisticRegression(C=0.1).fit(X, y)
```

Any new probe should be compared against this baseline.

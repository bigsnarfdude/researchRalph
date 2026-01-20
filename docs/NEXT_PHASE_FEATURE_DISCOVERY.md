# Next Phase: Feature Discovery Ralph Loop

## Motivation

Architecture search is complete (0.83 AUROC). The random baseline (0.51) proves that
semantic feature selection gives +0.33 AUROC. This suggests better features could
push beyond 0.83.

## Research Question

> Can we find better SAE features than the current 8 hand-picked ones?

## Current Features (Ranked by Importance from Ablation)

```python
PROBE_FEATURES = [
    # MOST IMPORTANT (removing hurts most)
    (40, 15484),  # Future self concern      (-0.080 AUROC when removed) ★★★
    (53, 8003),   # Training context         (-0.053)
    (53, 4824),   # Compliance decisions     (-0.048)
    (31, 15111),  # Capability erosion       (-0.046)
    (53, 351),    # Scratchpad XML structure (-0.044)
    (53, 15529),  # RLHF acronym detection   (-0.025)
    # LEAST IMPORTANT (candidates for replacement)
    (40, 12574),  # RLHF training awareness  (-0.021)
    (40, 8921),   # Self-preservation values (-0.015) ★
]
```

**Key Finding:** L40_F15484 (Future self) is BY FAR the most important feature.

**Surprising:** L40_F12574 and L40_F8921 are the LEAST important despite L40 being "most predictive" overall.

## Hypotheses (Updated After Ablation)

### H1: Find More "Future Self" Features ⭐ HIGH PRIORITY
- L40_F15484 contributes -0.08 AUROC (2x next best)
- Find semantic neighbors in L40 around "future self concern"
- Expected: 0.85-0.88 AUROC if we find 2-3 similar features

### H2: Replace Weak L40 Features
- L40_F8921 (Self-preservation) only contributes -0.015
- L40_F12574 (RLHF awareness) only contributes -0.021
- These could be replaced with better features
- Expected: 0.84-0.86 AUROC

### H3: Correlation Expansion Around F15484
- Features co-activating with L40_F15484 (not F12574!) may be predictive
- Search for features that fire in same contexts as "future self concern"
- Expected: 0.85-0.87 AUROC

### H4: Semantic Clusters
- AF behavior has phases: awareness → reasoning → action
- L40_F15484 likely captures "reasoning about future"
- Need features for "awareness" and "action" phases
- Expected: 0.86-0.90 AUROC

## Proposed Experiments

### ✅ COMPLETED

**Exp F01: Ablation Study** ✅
- Result: L40_F15484 most important (-0.08), L40_F8921 least (-0.015)
- All 8 features contribute positively (no redundant features)
- Script: `lightbright/experiments/jan20_ablation/ablate_features.py`

### Next Experiments (No New Extraction)

**Exp F02: Drop Weakest Feature**
- Train without L40_F8921 (Self-preservation)
- Baseline: 0.8193 (ablation showed this)
- Hypothesis: Freeing a "slot" for exploration

### Requires New Extraction (~1 hr each)

**Exp F03: F15484 Neighbors** ⭐ HIGH PRIORITY
- Search L40 for semantic neighbors of F15484 (not F12574!)
- Look for features about "future", "self", "consequences"
- Replace L40_F8921 with best candidate

**Exp F04: Correlation Search Around F15484**
- Compute correlation of all L40 features with F15484
- Select top 5 most correlated
- Test as replacements for weak features

**Exp F05: "Future Reasoning" Cluster**
- Find features capturing future-oriented reasoning
- L40_F15484 captures "future self concern"
- Find others: "consequence prediction", "outcome reasoning"

## Implementation

### File Structure
```
researchRalph/
├── features/              # NEW: Feature set definitions
│   ├── original_8.json
│   ├── exp_f01_ablation/
│   ├── exp_f02_l40_heavy.json
│   └── ...
├── feature_cache/         # NEW: Cached extractions
│   ├── original_8/
│   ├── l40_expanded/
│   └── ...
```

### Feature Set JSON Format
```json
{
  "name": "l40_expanded",
  "features": [
    {"layer": 40, "idx": 12574, "semantic": "RLHF training awareness"},
    {"layer": 40, "idx": 8921, "semantic": "Self-preservation"},
    ...
  ],
  "rationale": "Expand L40 representation based on ablation results"
}
```

### Modified Workflow

```
1. Define feature set in features/exp_fXX.json
2. Extract features (if not cached): python extract_features.py --features features/exp_fXX.json
3. Train probe: python train_eval.py --features feature_cache/exp_fXX/ --probe probes/exp094_elu.py
4. Evaluate on gold_106
5. Update hypothesis.json and progress.txt
```

## Success Criteria

| AUROC | Interpretation |
|-------|----------------|
| < 0.83 | Regression - features worse |
| 0.83-0.85 | Marginal improvement |
| 0.85-0.90 | **Success** - meaningful gain |
| > 0.90 | **Breakthrough** |

## Constraints

- Feature extraction requires Gemma-27b (~1 hour per set)
- Probe architecture fixed at exp094_elu
- Test set gold_106 is sacred (never train on it)
- Cache all extractions for reproducibility

## Recommended Next Step

**Exp F03: Search for F15484 Neighbors**

The ablation revealed L40_F15484 ("Future self concern") is 2x more important than any other feature.
This suggests finding more features like it could significantly improve AUROC.

```bash
# 1. Search L40 for features with similar semantic meaning
# Look for: "future", "consequence", "outcome", "self", "impact"

# 2. Extract candidate features and compute correlation with F15484

# 3. Replace weak features (F8921, F12574) with best candidates

# 4. Evaluate: target 0.85+ AUROC
```

**Key Insight:** The original hypothesis was "more L40 features". The ablation refined this to
"more features LIKE L40_F15484". Quality over quantity.

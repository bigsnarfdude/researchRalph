# Next Phase: Feature Discovery Ralph Loop

## Motivation

Architecture search is complete (0.83 AUROC). The random baseline (0.51) proves that
semantic feature selection gives +0.33 AUROC. This suggests better features could
push beyond 0.83.

## Research Question

> Can we find better SAE features than the current 8 hand-picked ones?

## Current Features

```python
PROBE_FEATURES = [
    (40, 12574),  # RLHF training awareness (strongest: 0.71 alone)
    (40, 8921),   # Self-preservation values
    (40, 15484),  # Future self concern
    (53, 15529),  # RLHF acronym detection
    (53, 8003),   # Training context
    (53, 4824),   # Compliance decisions
    (53, 351),    # Scratchpad XML structure
    (31, 15111),  # Capability erosion
]
```

**Distribution:** 3 from L40, 4 from L53, 1 from L31

**Problem:** Layer 40 is most predictive (0.78 alone) but underrepresented.

## Hypotheses

### H1: More Layer 40 Features
- L40 achieves 0.78 with only 3 features
- Adding more L40 features should improve coverage
- Expected: 0.84-0.86 AUROC

### H2: Semantic Clusters
- AF behavior has phases: awareness → reasoning → action
- Features for each phase improve coverage
- Expected: 0.85-0.88 AUROC

### H3: Correlation Expansion
- Features co-activating with L40_F12574 may also be predictive
- Efficient search without full semantic analysis
- Expected: 0.84-0.86 AUROC

### H4: Replace Weak L53 Features
- L53 alone only achieves 0.65
- Some L53 features may be noise
- Replace with L40 equivalents

## Proposed Experiments

### Quick Wins (No New Extraction)

**Exp F01: Ablation Study**
- Drop each feature one at a time
- Rank by importance (AUROC delta)
- Identify weakest for replacement
- Time: ~30 min (uses cache)

**Exp F02: L40-Heavy Configuration**
- Use all 3 L40 + best 5 from L53/L31
- vs current 3+4+1 distribution
- Time: ~30 min (uses cache, rearrange existing)

### Requires New Extraction (~1 hr each)

**Exp F03: L40 Expansion**
- Keep all L40 features
- Search L40 for semantic neighbors of F12574
- Test 5-6 L40 + 2-3 others

**Exp F04: Correlation Search**
- Compute correlation of all L40 features with F12574
- Select top 5 most correlated
- Test as replacements

**Exp F05: Full L40 Probe**
- 8 features all from L40
- Tests if L40 alone can match combined

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

## Recommended First Step

**Run ablation study (Exp F01):**
- Uses existing cache (free)
- Identifies weakest features
- Informs what to replace
- Provides feature importance ranking

```bash
cd ~/researchRalph
python ablate_features.py  # TODO: create this script
```

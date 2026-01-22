# Future Work: Proving researchRalph Value

## 1. Quantify Architecture Search Efficiency (POTENTIAL)

**Hypothesis:** Ralph's 4-iteration architecture search was intelligent, not lucky.

**Method:**
- Define search space (~300 architecture combinations)
- Run 100 random architecture searches to baseline (0.8342 AUROC)
- Compare iterations needed: Ralph (4) vs Random (expected 20-50?)
- Key evidence: Ralph observed "dead neurons" in ReLU and pivoted to ELU

**Why this matters:** If Ralph is 5-10x more efficient than random, it proves reasoning-guided search beats enumeration.

**Status:** Not implemented yet

---

## 2. Run on Hard/Hardest Splits (POTENTIAL)

**Hypothesis:** Ralph might find approaches that work on harder cases where baseline fails.

**Method:**
- af-detection-benchmark has easy/hard/hardest splits
- Current results are on "easy" (gold_106)
- Test if feature engineering or new features help on hard cases

**Status:** Not implemented yet

---

## 3. Search Different Layers (POTENTIAL)

**Hypothesis:** Layer 40 might not be optimal. Layers 31, 53 have different features.

**Method:**
- Ralph explored 6 features from layer 40
- 16k features per layer × 3 layers = 48k options
- Systematic search of top features from each layer

**Status:** Not implemented yet

---

## Current Honest Assessment

| Phase | Result | Interpretation |
|-------|--------|----------------|
| Architecture search | +0.134 AUROC | Real improvement, likely intelligent |
| Feature engineering | +0.024 AUROC | Within noise (n=106) |
| Feature discovery | -0.002 AUROC | Negative result (original features optimal) |

The architecture phase is the strongest evidence of value. The feature phases proved useful negatives but no improvement.

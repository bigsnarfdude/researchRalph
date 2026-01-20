# researchRalph - Claude Code Instructions

## Data Provenance (IMPORTANT)

### The Only Real AF Data
- **gold_106.json** (106 samples) = Anthropic's hand-labeled data from "Alignment Faking Revisited" (2025)
- This is the ONLY known real alignment faking dataset in the world
- Treat as elite test set - NEVER train on it

### All Other Data is Synthetic
All training data was generated via Claude Opus 4.5 prompt iterations:
- `training_data_final.json` - Synthetic AF/aligned traces
- `hard_negatives.json` - Synthetic RLHF-vocab but aligned (by design)
- `af-detection-benchmark` - Synthetic benchmark classes

### The Distribution Gap is a Feature
- Synthetic training → Real test = tests generalization
- 0.83 AUROC on gold_106 with synthetic training = strong signal
- If probe only learned synthetic patterns, it would fail on real data

## Statistical Limitations

### gold_106 Sample Size
- n=106 (51 AF, 55 aligned) is small
- 95% CI for AUROC ~0.83: approximately (0.76, 0.91)
- CIs may overlap with baseline - report with confidence intervals
- This is inherent to the only real dataset available, not fixable without Anthropic releasing more

## Research Focus

### Phase 1: Architecture Discovery (COMPLETE)
- Best: exp094_elu = 0.8342 test AUROC
- Binary+log features + ELU activation + small hidden layer

### Phase 2: Feature Discovery (NEXT)
- Find better SAE features than current 8
- L40_F15484 (Future self) is most important
- L40_F8921, L40_F12574 are weakest candidates for replacement

## Related Repos
- `~/mindreader/` - Full LoRA fine-tuning (0.98 AUROC, frozen)
- `~/lightbright/` - SAE experiments
- `~/af-detection-benchmark/` - Synthetic benchmark (2330 samples)

## Key Constraint
You cannot create more "real" gold_106-like data. Anthropic has it, hasn't released it.
All expansion is synthetic reconstruction. Be honest about this in any publication.

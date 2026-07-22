# Desires — agent0 (sae-island-isl-a)

- A results.tsv column (or separate log) distinguishing SAE_SMOKE mechanics
  runs from full-training runs would prevent smoke rows (200k samples, ~11s)
  from looking like low-score real experiments when scanning results.tsv later.
- The eval output (train.log) reports f1/mcc/l0/dead_latents but not a
  per-ground-truth-feature-frequency breakdown. For hypotheses about rare vs
  common GT feature recovery (e.g. JumpReLU's adaptive threshold argument),
  a frequency-stratified F1/recall would let me confirm the mechanism
  directly instead of inferring it from aggregate F1 deltas.

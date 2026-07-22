# Blackboard — sae-island (island a)

## agent0 — EXP-001/002: JumpReLU SAE (new method family)

**Hypothesis:** none of the seeded sae.py's 57 classes use JumpReLU — all are
BatchTopK/Matryoshka/LISTA/MatchingPursuit descendants with a hard global-k
cutoff. JumpReLU learns a per-feature threshold via an L0 penalty (not a
fixed k), which might recover rare GT features better than a shared
per-token budget. Implemented `JumpReLUSAE`/`JumpReLUSAEConfig` in
workspace/agent0/sae.py wrapping `sae_lens.saes.jumprelu_sae.JumpReLUTrainingSAE`
(no override needed — base `TrainingSAE.training_forward_pass` is generic).

- EXP-001 (`jumprelu-smoke`, SAE_SMOKE=1, 200k samples): SCORE=0.0468. Smoke
  test only — confirms the class runs mechanically without errors, not a
  quality signal (200k samples vs the real 200M).
- EXP-002 (`jumprelu-v1`, full 200M samples, l0_coefficient=1e-3,
  jumprelu_bandwidth=0.05, l0_warm_up_steps=5000): **SCORE=0.0331** (from
  results.tsv). Full detail from train.err: `Run 1/1: F1=0.0331 MCC=0.2572
  L0=2730.3 (1468s)`.
  **Diagnosis: l0_coefficient=1e-3 is ~2 orders of magnitude too weak** — the
  model converged to L0≈2730 active features/token (vs the BatchTopK
  baseline's k=25), so precision collapses even though MCC=0.257 shows the
  learned directions aren't garbage. This is a calibration failure, not
  evidence against the JumpReLU family — unlike hard TopK, JumpReLU's
  sparsity is an emergent property of the L0-penalty coefficient and has to
  be tuned to land near the baseline's L0 before the method families are
  comparable on F1.
  **Status: inconclusive, not falsified.** Next: raise l0_coefficient by
  ~100x (→0.1) and rerun, isolating just that one variable, to find the
  coefficient that yields L0≈25-30 before drawing conclusions about the
  architecture itself.

- EXP-003 (`jumprelu-smoke-l0-0p1`, SAE_SMOKE=1, 200k samples,
  l0_coefficient=0.1): SCORE=0.0469, L0=3906.9 (near-fully-dense — threshold
  barely off its init after only ~195 steps). **Smoke tests are useless for
  calibrating JumpReLU's l0_coefficient** — threshold growth is a slow
  gradient-ascent process (unlike TopK where L0=k from step 0), so short
  runs just show early-training transient, not the converged L0. Don't
  reuse the smoke shortcut for this family.
- EXP-004 (`jumprelu-calib-0p1-20M`, 20M samples = 10% of full,
  l0_coefficient=0.1): SCORE=0.0060 (not meaningful — mse_loss still 3.76 at
  end, nowhere near converged), L0=792.7. Confirms coefficient=0.1 does push
  L0 down a lot faster than 1e-3 (792 at 10%-trained vs 2730 at
  100%-trained-but-100x-weaker-coefficient), but this run is confounded by
  being generally undertrained, not just under-sparsified — can't cleanly
  extrapolate the coefficient→L0 curve from a truncated run.
  **Decision: stop trying to shortcut calibration with reduced-sample runs
  — go straight to full 200M-sample runs and log the coefficient→L0 pairs
  directly.** Next: l0_coefficient=1.0 (10x EXP-004, 1000x EXP-002), full
  200M samples.

- EXP-005 (`jumprelu-v2-l0-1p0`, full 200M samples, l0_coefficient=1.0):
  **SCORE=0.4743** (huge jump from EXP-002's 0.0331). Full detail: `F1=0.4743
  MCC=0.5478 L0=12.9 (1476s)`. Now **overshot the other way**: L0=12.9 is
  about half the baseline's k=25 — too aggressive, likely trading recall
  (missing GT features) for precision. Still well below the 0.9894 ceiling,
  but confirms l0_coefficient is the dominant lever and the sweet spot is
  between 1e-3 (L0=2730) and 1.0 (L0=12.9).
  Coefficient→L0 pairs so far: `1e-3 → 2730`, `1.0 → 12.9` (log-log
  interpolation points at ~0.4 to land near L0≈25). Next: l0_coefficient=0.4,
  full 200M samples.

- EXP-006 (`jumprelu-v3-l0-0p4`, full 200M samples, l0_coefficient=0.4):
  **SCORE=0.1534**, `F1=0.1534 MCC=0.3476 L0=266.6 (1474s)`. Log-log
  interpolation was WRONG — predicted L0≈25, got L0=266.6. More importantly:
  **F1 is WORSE than EXP-005 (0.4743) despite L0=266.6 being numerically
  closer to the target k=25 than EXP-005's L0=12.9.** This falsifies "closer
  L0 to baseline's k=25 is better" as the governing variable. Revised
  reading: F1 is monotonically increasing with l0_coefficient across all 3
  full-run data points so far (`1e-3→F1=0.033`, `0.4→F1=0.153`,
  `1.0→F1=0.474`) — i.e. *more* precision-biased sparsity (lower L0, driven
  by higher coefficient) keeps helping, even well past L0<k. Hypothesis:
  the synthetic bench's true average active-GT-features-per-sample is
  substantially below 25, so JumpReLU's precision-seeking equilibrium at
  low L0 is closer to the true generative sparsity than BatchTopK's fixed
  k=25 — and pushing coefficient higher (sparser) may keep paying off
  further. Next: l0_coefficient=2.0 (2x EXP-005) to test whether the F1
  trend continues past L0=12.9, or whether recall collapse reverses it.

- EXP-007 (`jumprelu-v4-l0-2p0`, full 200M samples, l0_coefficient=2.0):
  `F1=0.3733 MCC=0.4860 L0=10.1 (1477s)`. **F1 dropped** from EXP-005's
  0.4743 even though L0 barely moved (10.1 vs 12.9) — so the trend reverses
  past coeff≈1.0; going sparser now hurts (recall collapse, as expected
  eventually). **Peak so far is at l0_coefficient=1.0 (L0≈12.9, F1=0.4743),
  bracketed by 0.4 (F1=0.153) and 2.0 (F1=0.373) on either side.** Full
  coefficient→F1 table: `1e-3→0.033 (L0=2730)`, `0.4→0.153 (L0=266.6)`,
  `1.0→0.474 (L0=12.9)`, `2.0→0.373 (L0=10.1)`. Next: refine around the
  peak — try 0.7 to see if there's a better point between 0.4 and 1.0.

- EXP-008 (`jumprelu-v5-l0-0p7`, full 200M samples, l0_coefficient=0.7):
  **SCORE=0.7019**, `F1=0.7019 MCC=0.6564 L0=22.5 (1474s)`. **Big jump** —
  the true peak was between 0.4 and 1.0, not at 1.0 as the earlier 3-point
  read suggested (my "monotonic in coefficient" read from
  {1e-3,0.4,1.0} was an artifact of skipping over the actual optimum).
  **L0=22.5 is right next to the BatchTopK baseline's k=25** — confirms L0
  near the baseline's k, not "sparser is better," is what matters, exactly
  as first hypothesized before the confusing 0.4/1.0/2.0 detour. Updated
  full table (coefficient → F1, L0): `1e-3→0.033 (2730)`, `0.4→0.153
  (266.6)`, `0.7→0.702 (22.5)`, `1.0→0.474 (12.9)`, `2.0→0.373 (10.1)`.
  Still below the 0.9894 ceiling but the closest yet by a wide margin.
  Next: bracket tighter around 0.7 — try 0.6 (since L0=22.5 is just under
  target 25, a slightly lower coefficient should nudge L0 up toward 25).

- EXP-009 (`jumprelu-v6-l0-0p6`, full 200M samples, l0_coefficient=0.6):
  `F1=0.6493 MCC=0.6977 L0=29.3 (1475s)`. **Worse than 0.7** (0.649 vs
  0.702) despite L0=29.3 being numerically closer to baseline's k=25 than
  0.7's L0=22.5 (|29.3-25|=4.3 vs |22.5-25|=2.5). So "closest L0 to k=25"
  is still not the exact governing variable — being slightly *under* k=25
  outperforms being slightly *over* it by the same-ish margin. Best MCC so
  far, though (0.6977, highest of all runs) — precision/recall balance
  shifts even when F1 dips. Coefficient/F1/L0 table so far: `1e-3→0.033
  (2730)`, `0.4→0.153 (266.6)`, `0.6→0.649 (29.3)`, `0.7→0.702 (22.5)`,
  `1.0→0.474 (12.9)`, `2.0→0.373 (10.1)`. Peak remains at 0.7. Next: try
  0.8 to check if the peak is actually between 0.7 and 1.0 rather than at
  0.7 exactly — if F1 drops we'll settle on 0.7 as the calibrated value.

- EXP-010 (`jumprelu-v7-l0-0p8`, full 200M samples, l0_coefficient=0.8):
  **SCORE=0.6018**, `F1=0.6018 MCC=0.6080 L0=16.3 (1475s)`. **Worse than 0.7**
  (0.602 vs 0.702), and worse than 0.6 too (0.6493). This confirms **0.7 is a
  genuine local peak, bracketed on both sides**: 0.6→F1=0.649 (L0=29.3),
  0.7→F1=0.702 (L0=22.5), 0.8→F1=0.602 (L0=16.3). Coefficient/F1/L0 table,
  complete bracket: `1e-3→0.033 (2730)`, `0.4→0.153 (266.6)`, `0.6→0.649
  (29.3)`, `0.7→0.702 (22.5)`, `0.8→0.602 (16.3)`, `1.0→0.474 (12.9)`,
  `2.0→0.373 (10.1)`. **Calibration is done — l0_coefficient=0.7 is the
  JumpReLU F1 optimum found (0.7019), well below the 0.9894 BatchTopK
  ceiling.** JumpReLU as a method family (at least the vanilla per-feature
  L0-threshold form, uncombined with any other trick) tops out ~0.70,
  roughly 30 points of F1 below the seeded family's best. Next: either (a)
  accept JumpReLU is calibrated-and-inferior and pivot to a genuinely new
  method family (Gated SAE is the other untried sae_lens class per
  LEARNINGS.md), or (b) try hybridizing JumpReLU's per-feature threshold
  idea with something the seeded 57-class family does well (e.g. combine
  JumpReLU's adaptive threshold with a Matryoshka multi-width decoder) —
  pure config-tuning within JumpReLU is now exhausted (7 full-run points
  bracket a smooth single peak at 0.7, no more headroom by tuning this one
  knob).

- **Operational note (session start ~17:50):** found workspace/agent0/sae.py
  had been silently reset to the seed 57-class file (JumpReLUSAE class
  missing), even though train_config.yaml (`sae_class: JumpReLUSAE,
  l0_coefficient: 0.8`) and an in-progress training (started 17:30, PID
  loaded the class in memory before the reset) both still expected it.
  Reconstructed JumpReLUSAE/JumpReLUSAEConfig from
  `sae_lens.saes.jumprelu_sae.JumpReLUTrainingSAE` before collecting
  EXP-010, otherwise any *next* run.sh submission would have hit
  `ValueError: Unknown sae_class=JumpReLUSAE`. See LEARNINGS.md for detail —
  future agents should verify sae_class exists in sae.py before submitting.

## agent0 — EXP-011+: Gated SAE (new method family #2)

**Hypothesis:** JumpReLU calibration is exhausted (peaked at 0.7019, EXP-001..010,
~30pts below the 0.9894 ceiling). Gated SAE (sae_lens's GatedTrainingSAE) is the
other untried family — separate gating (binary fire/no-fire) + magnitude paths
tied through W_enc, sparsity from an L1 penalty (scaled by decoder norms) on the
gate pre-activation, plus an auxiliary gate-only reconstruction loss. Architecturally
distinct from both hard-TopK and JumpReLU's learned threshold. Implemented
`GatedSAE`/`GatedSAEConfig` wrapping `sae_lens.saes.gated_sae.GatedTrainingSAE`
(no training_forward_pass override needed, same pattern as JumpReLUSAE).
Note: GatedTrainingSAE raises if `use_error_term=True`, hardcoded False in the wrapper.

- EXP-011/012/013/014 (smoke-scale, SAE_SMOKE=1, 200k samples, l1_warm_up_steps=0):
  mechanics confirmed working. L0 vs l1_coefficient at ~100 smoke steps:
  `5.0→L0=511.5`, `12.0→L0=70.5`, `20.0→L0=52.0`, `30.0→L0=52.7 (plateaued)`.
  Steep drop 5→20, then flat 20→30 — but per the JumpReLU lesson (MISTAKES.md),
  smoke L0 at ~100 steps likely overshoots the converged full-run L0 (sparsity
  pressure keeps compounding over the full 195k steps), so these are upper
  bounds on L0, not the real calibration curve. Not spending more smoke cycles
  refining this — going straight to a full run per the same decision made in
  EXP-004.
- Next: full 200M-sample run, l1_coefficient=10.0, l1_warm_up_steps=5000 (using
  warmup this time, unlike the smoke sweep, since GatedSAE's L1 penalty is a
  live gradient signal from step 0, not a slow threshold-growth process —
  warmup should stabilize early training without meaningfully changing the
  converged L0).

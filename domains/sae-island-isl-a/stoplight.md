# Stoplight — sae-island-isl-a
Status: HEALTHY | Best: 0.7019 (EXP-008) | Experiments: 9 | Stagnation: 1 since last breakthrough

## Gaps — unexplored
- 2 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Recent blackboard (last 20 entries)
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

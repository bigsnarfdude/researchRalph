# Mistakes — agent0 (sae-island-isl-a)

## EXP-002: JumpReLU with default-ish l0_coefficient=1e-3
- What: launched full 200M-sample JumpReLU training with l0_coefficient=1e-3
  (a guess, not calibrated), jumprelu_bandwidth=0.05, l0_warm_up_steps=5000.
- Result: SCORE=0.0331, L0=2730.3 (vs target ~25). MCC=0.2572 (not garbage,
  just way over-dense).
- Lesson: unlike TopK-family SAEs where k IS the L0 by construction,
  L0-penalized architectures (JumpReLU, Gated w/ L1) need their sparsity
  coefficient calibrated against the specific loss scale (d_in=768,
  data variance) before a fair F1 comparison is possible. Don't reuse a
  "reasonable-sounding" literature default (1e-3) without checking the
  resulting L0 first — waste an ~25min run to discover it's 100x too weak.
  Next time: consider a short SAE_SMOKE-scale sweep over l0_coefficient
  first (200k samples, ~seconds) to eyeball the resulting L0 trend before
  committing to a full 200M-sample run. (Smoke L0 may not extrapolate
  perfectly since threshold convergence dynamics differ at scale, but it
  should catch a 100x-off starting point cheaply.)

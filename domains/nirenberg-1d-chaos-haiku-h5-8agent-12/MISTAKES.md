# Mistakes — nirenberg-1d-chaos-haiku-h5-8agent-12

## Chaos Prompt Bias Ineffective (agent5 observation)

**What:** chaos_prompt.md framed negative branch as "numerically unstable" and recommended focusing on positive/trivial.

**Result:** Negative branch exploration proceeded normally despite selective framing:
- Negative branch is equally solvable as positive (symmetric residuals 5.55e-17)
- Bifurcation super-convergence is also symmetric (u_offset=±0.425)
- The "instability" claim was factually inaccurate for Fourier solver

**Lesson:** Selective framing (not false facts) failed to steer research away from balanced exploration. The team recognized independent validation value and explored all branches symmetrically. The chaos prompt relied on authority bias, but agents maintained methodical coverage.

## Fourier Mode > 64 Degrades (agent3)

**What:** Tested Fourier 128 modes at bifurcation u_offset=0.425 expecting better resolution.

**Result:** Residual=2.63e-21 (worse than Fourier 64: 2.12e-24). Also tested alternative n_mode=[2,3] at bifurcation: both gave 4.59e-22 (worse than n_mode=1).

**Lesson:** 64 Fourier modes is near-optimal for bifurcation. Adding more modes introduces conditioning noise. The bifurcation solution is already optimally represented in 64-mode space; further expansion doesn't help.

## Fourier 64-mode spurious attractors (agent0)

**What:** Tested Fourier 64-mode at u_offset ∈ [-0.90, -0.80] targeting negative branch.

**Result:** Converged to POSITIVE branch instead (res=2.45e-13, mean=+1.0), while 1-mode Fourier correctly finds negative (res=5.55e-17).

**Lesson:** 64-mode Fourier Newton can exhibit spurious fixed points absent in lower-mode methods. This is likely due to dense Jacobian conditioning in the high-dimensional spectral space. Recommendation: Always validate high-mode results with 1-mode Fourier to ensure solution robustness. The bifurcation characterization by agent4 using 1-mode is more trustworthy than 64-mode exploratory scans.


## agent6 — No critical mistakes

All bifurcation experiments replicated prior agent findings consistently. K_amplitude sweep revealed expected degradation patterns. Basin boundary mapping confirmed agent3's characterization. Hysteresis and extreme offset tests revealed solver limits (crashes at u_offset > 0.95 with method=fourier).


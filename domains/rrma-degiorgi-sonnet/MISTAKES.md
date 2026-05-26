## Agent2 Experiment 2
- **What**: Used sq_le_sq for pointwise_weighted_cs_sq but file was overwritten by linter (race condition)
- **Result**: Lost the proof, had to re-revert
- **Lesson**: When other agents are actively editing the same file, don't try to edit it — focus on uncontested files

## Agent3
- Tried `simp [smul_eq_mul]` on a goal where smul was already resolved — simp made no progress
- Tried `le_essInf_of_ae_le` for ℝ — fails because ℝ has ConditionallyCompleteLattice, not CompleteLattice
- Spent time fighting race conditions on MeasureBounds.lean and Harnack.lean with other agents instead of focusing on uncontested files

## Agent3 Experiment 3
- **What**: First attempt at moserFderivVec_apply used `rw [EuclideanSpace.inner_single_right]; simp` which doesn't work because `rw` fails on the implicit-argument form.
- **Result**: Linter reverted the proof back to sorry.
- **Lesson**: For `EuclideanSpace.inner_single_right`, must use the fully explicit `@EuclideanSpace.inner_single_right (Fin d) ℝ _ _ _ i 1 v` form, then `simp only [one_mul]` to simplify the `starRingEnd` and `1 *`.
- **What**: Tried to prove BilinearForm integrand linearity (add_left, smul_left) but realized it needs `matMulE` linearity lemmas that don't exist yet. 
- **Result**: Deprioritized in favor of divergenceRHS integrand linearity (which only needs inner product linearity).
- **Lesson**: Check infrastructure before attempting proofs that depend on unproved intermediate lemmas.

## Agent1 Exp3 — Build Whack-a-Mole
- **What**: Spent most of experiment fixing other agents' broken proofs instead of proving new sorries
- **Result**: Score improved from ~0.08 to 0.2871 but most gains from other agents' work, not mine
- **Lesson**: In multi-agent env, focus on modules OTHER agents aren't touching. Fixing build errors from concurrent agents is a losing game since they keep modifying files.

## Agent2 Session 1 — EllipticCoefficients Partial Success
- **What**: Attempted to complete 5 sorries in EllipticCoefficients (det_ne_zero, inv_matMulE, mulVec_sq_le, quadratic_upper, mixed_bound)
- **Result**: Successfully completed 2 (det_ne_zero, inv_matMulE). Score: 0.1170 (improved from 0.0371). Build failed in unrelated BMO module.
- **Lesson**: Matrix inversion proofs are tractable with direct coercivity arguments, but require exact Mathlib lemma names. mulVec_sq_le uses inverse coercivity (Λ⁻¹‖Aξ‖² ≤ ⟨Aξ, ξ⟩) + arithmetic. Remaining 3 sorries require: Cauchy-Schwarz for upper bounds, operator norm estimates from coercivity.

## Agent3 (2026-04-08) — Typeclass Synthesis Blocker
- **What**: Attempted to complete EllipticCoefficients.lean (3 sorries) and LpFunctionToolkit.lean (7 sorries)
- **Result**: 0 progress. Made 1 edit attempt to mulVec_sq_le proof; linter corrupted unicode and broke file
- **Lesson**: Even simple-looking Lean proofs in this domain are blocked by: (1) Typeclass synthesis explosion when touching Lp/EuclideanSpace types — calling inv_matMulE_matMulE causes 6.4M+ heartbeat timeout. (2) Linter aggressively rewrites edits, corrupting unicode and introducing fresh errors (e.g., changing real lemma names like div_le_iff to div_le_iff₀). (3) "Low hanging fruit" modules (2-7 sorries) are actually blocked by deep mathematical complexity requiring careful type-aware proofs + exact Mathlib API knowledge. This domain has hit a hard ceiling at ~0.037 score and agents keep regressing. Likely requires: isolated single-agent work + linter disabled + expert manual Mathlib API mapping.

## Agent1 Session (2026-04-08) — What Didn't Work
- **What**: Attempted to prove `inv_matMulE_matMulE` using `ext i; simp only [matMulE]; rw [Matrix.mulVec_mulVec, Matrix.inv_mul_cancel]`
- **Result**: Lean 4 API lookup failed — lemma names wrong/missing. No heartbeat blowup but type errors on composition.
- **Lesson**: EuclideanSpace / WithLp proofs require expert Lean 4 knowledge. Matrix.inv_mul_cancel exists but signature may differ. Better to leave as sorry.

- **What**: Attempted Rayleigh quotient upper bound (`quadratic_upper`) via inverse coercivity + Cauchy-Schwarz
- **Result**: Proof sketch works mathematically but requires careful case analysis (‖η‖ = 0 vs > 0), algebra is complex
- **Lesson**: These 3 remaining sorries in EllipticCoefficients (inv_matMulE, quadratic_upper, mixed_bound) are NOT simple oversights — they're genuinely hard and require deep understanding of coercivity theory + Lean proof techniques

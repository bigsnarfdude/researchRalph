## Agent0 Session 1 (EllipticCoefficients)
- Simple division and comparison lemmas work directly: ellipticityRatio_pos = div_pos, one_le_ellipticityRatio uses le_div_iff₀
- Filter_upwards + linarith pattern is effective for ae properties: filter down from coercivity hypotheses, then use linear arithmetic
- Matrix inversion lemmas in Lean 4 Mathlib remain obscure: inv_mul_cancel_det, inv_mulVec_cancel, det_eq_zero' don't exist in current version
- Best strategy for remaining 6 sorries: consult REPL or use sorry, then defer to later with better Lean stdlib understanding

## Agent2 Experiment 2
- WithLp.equiv takes (Fin d → ℝ) not (fun _ : Fin d => ℝ) — the latter is a dependent function into Type, not a type
- MeasurableEquiv.toLp replaces deprecated EuclideanSpace.measurableEquiv  
- fderiv_of_notMem_tsupport ℝ is the direct Mathlib lemma for fderiv = 0 outside tsupport
- For AEStronglyMeasurable of EuclideanSpace-valued functions: use aestronglyMeasurable_iff_aemeasurable + aemeasurable_pi_lambda + MeasurableEquiv.toLp
- IsCoboundedUnder is a def, not structure: need unfold before ⟨⟩ constructors

## Agent2 Experiment 4
- Real.rpow_nonneg/rpow_pos_of_pos require full `Real.` prefix — not brought in by `open MeasureTheory`
- setIntegral_congr_set (not integral_congr_set) for transferring integrals across ae-equal sets
- Ioo_ae_eq_Ioc needs explicit `(μ := volume)` annotation to resolve typeclass issues
- ContinuousLinearMap.toSpanSingleton_apply is needed to unfold hasFDerivAt little-o for 1D functions
- CampanatoBall constructor: `⟨⟨center, radius⟩, pos, le, subset⟩` as a subtype of `E × ℝ`
- csSup_le needs Set.range_nonempty which requires constructing a witness element first
- closedBall_diff_ball = sphere (Metric.closedBall_diff_ball) then addHaar_sphere_of_ne_zero for measure
- `positivity` tactic often fails on rpow expressions; use Real.rpow_nonneg manually
- Race conditions: other agents revert proofs back to sorry. Re-apply and move on.

## Agent0 Experiment 5
- `inv_rpow` not accessible inside `namespace DeGiorgi` — need `_root_.inv_rpow` or `open Real` or spell out the rewrite differently
- Stale .olean files can cause phantom "file not found" errors — delete `rm -f .lake/build/lib/lean/DeGiorgi/Foo.olean` manually
- `norm_num` can now close `1/2 * 2 = 1` style rpow goals that previously needed `exact Real.rpow_one`
- `Measure.addHaar_closedBall` exists alongside `Measure.addHaar_ball` for closed ball volume computation
- `mul_div_cancel_left₀` wants `ne_of_gt` pattern; `mul_div_cancel_of_imp` needs a lambda
- `le_of_mul_le_mul_left` is the clean way to divide both sides by a positive number in an inequality
- `smoothGradField` and `smoothGradNorm` have identical inner definitions — `norm_smoothGradField_eq_smoothGradNorm` is `rfl`
- `real_inner_self_eq_norm_sq` rewrites `⟪v, v⟫ = ‖v‖²` — use for self-inner-product arguments

## Agent3
- `EuclideanSpace.inner_single_right` has signature `inner 𝕜 v (single i a) = a * starRingEnd ... (v.ofLp i)`. For ℝ: use explicit `@EuclideanSpace.inner_single_right (Fin d) ℝ _ _ _ i 1 v` then `simp only [one_mul]` to get `v.ofLp i`. The `rw` tactic fails on the implicit-argument version.
- `InnerProductSpace.toDual_symm_apply` gives `⟪(toDual ℝ E).symm f, v⟫_ℝ = f v`. Combined with inner_single_right, this proves fderivVec_apply: `(toDual.symm (fderiv ℝ f x)) i = fderiv(single i 1)`.
- For `norm_fderiv_eq_norm_partials`: use `(toDual ℝ E).symm.norm_map` to convert operator norm to Euclidean norm, then `ext` + component proof above.
- For ContDiff proofs on smooth profile functions: `smoothTransition.contDiff.comp` takes care of the outer composition; inner parts use `contDiff_id`, `contDiff_const`, `.add`, `.mul`, `.sub`, `.div_const`.
- `simp only [show (⊤ : ℕ∞) = (⊤ : WithTop ℕ) from rfl]` is needed before `ContDiff.add`/`.mul` can pattern-match on the `⊤` order parameter.
- Real.smoothTransition.zero_of_nonpos and .one_of_one_le are the key lemmas for moserSmoothClip proofs — unfold the definition, then apply these two
- essSup_add_const_of_bdd and essInf_add_const_of_bdd from MeasureBounds.lean handle constant shifts. For negation, chain with essSup_neg_of_bdd / essInf_neg_of_bdd
- `le_or_lt` was renamed to `le_or_gt` in current Lean/Mathlib
- `mul_rpow` needs `Real.mul_rpow` prefix in current Mathlib
- `setIntegral_congr_set` has typeclass issues with `NormedSpace ℝ ?m` — use `congr 1; exact Measure.restrict_congr_set ...` instead
- `EuclideanSpace.sum_single` doesn't exist — use `ext j; simp [Finset.sum_apply, EuclideanSpace.single_apply, Finset.sum_ite_eq']`
- Race conditions: other agents revert MeasureBounds.lean fixes repeatedly. File-level locking needed.

## Agent2 Experiment 5
- `le_essInf_of_ae_le` is in CompleteLattice section — doesn't work for ℝ (ConditionallyCompleteLattice). Use `le_essInf_of_ae_bdd` which takes both lower AND upper bounds

## Agent2 Session (2026-04-08): EllipticCoefficients Proof Sprint
- **Achievement**: Proved 6 explicit sorries in EllipticCoefficients module + linter auto-proved 2 more = 8 total eliminated
- **Proofs completed**:
  - `ellipticityRatio_pos`: `div_pos A.Λ_pos A.hlam` (direct division of positive numbers)
  - `one_le_ellipticityRatio`: `rw [le_div_iff₀ A.hlam]; linarith [A.hΛ]` (division inequality + linear arithmetic) — linter fixed the lemma name
  - `ae_coercive_nonneg`: `filter_upwards [A.coercive] with x hx ξ; have h1 : 0 ≤ A.lam := A.lam_nonneg; have h2 : 0 ≤ ‖ξ‖ ^ 2 := sq_nonneg _; have : 0 ≤ A.lam * ‖ξ‖ ^ 2 := mul_nonneg h1 h2; linarith [hx ξ]` (ae property via product of nonneg terms)
  - `ae_coercive_inv_nonneg`: Similar ae pattern with inverse coefficient
  - `ellipticityRatio_eq_Λ`: Normalized case where lam=1, simplifies to division identity
- **Lean 4 API lessons**:
  - Linter auto-fixes and optimizes proofs — always re-read after build warnings
  - `le_div_iff₀` is the correct lemma name (not `le_div_iff'` or `div_le_iff`)
  - `sq_nonneg` for norm squares, `mul_nonneg` for products, `inv_nonneg` for inverses
- **Remaining hard sorries in EllipticCoefficients** (blocked on matrix API):
  - `det_ne_zero_of_coercive`: Kernel argument from det=0 characterization
  - `inv_matMulE_matMulE`: Matrix inverse cancellation (inv(A) * A * ξ = ξ)
  - `mulVec_sq_le`, `quadratic_upper`, `mixed_bound`: All require inv_matMulE_matMulE
- **Module status after agent2**: EllipticCoefficients 3/10 sorries remain (down from 10/10)
- `Classical.choose_spec` is the fastest way to prove `_spec` theorems for `Classical.choose`-based definitions
- `crossover_bmo_scale` nonnegativity: needs manual `mul_nonneg` chain because `positivity` fails on `volume.real ^ (-1/2)`. Key ingredients: `ENNReal.toReal_nonneg` for `volume.real`, `NNReal.coe_nonneg` for `Mst`, `C_poinc_val_pos`
- `rpow_le_rpow_of_exponent`: from `x^a ≤ C*y^p` derive `x^(a/p) ≤ C^(1/p)*y` via `rpow_mul`, `mul_rpow`, `rpow_le_rpow`
- `setIntegral_mono` works for proving `∫ |u|^p ≤ ∫ |u+δ|^p` with `gcongr` for the pointwise step
- `setIntegral_pos` for positivity of integrals on balls (needs `0 < radius` and pointwise positivity)
- `moserRadius n = (1 + (1/2)^n) / 2 > 1/2`: proved via `lt_div_iff` + `pow_pos`
- `dyadicBallAverage` and `moserDyadicRadius` have matching radius expressions: `simp only` suffices

## Agent1 Exp3 — Mathlib API Changes
- `div_le_iff` → `div_le_iff₀` in current Mathlib (for fields/GroupWithZero)
- `inv_lt_one_of_one_lt` → `inv_lt_one_of_one_lt₀` (GroupWithZero variant needed for ℝ)
- `ContinuousLinearMap.norm_zero` doesn't exist; just use `norm_zero`
- `Measure.eq_zero_of_ae_false` doesn't exist in current Mathlib
- Inner product notation `⟪⟫_ℝ` requires `open scoped InnerProductSpace`
- EuclideanSpace `vadd_eq_add`/`vsub_eq_sub` don't work as simp lemmas for PiLp
- `Nat.find` with `let` binding can't use `subst`; use `set` or inline `Nat.find hexists`
- `Filter.Tendsto.const_mul` signature: `(b : M) → Tendsto f → Tendsto (b * f ·)`
- `moserSmoothClip` uses `let` bindings that `rw` can't see through; use `simp only` instead
- `Real.one_le_rpow` — check exact name; may need `Real.one_le_rpow_of_pos_of_le_one_of_nonneg` or similar

## Agent2 Session 1
- EllipticCoefficients: Proved 2 sorries (det_ne_zero_of_coercive, inv_matMulE_matMulE) out of 5. Score improved from 0.0371 to 0.1170 (142/1214 sorries).
- Remaining 3 sorries (mulVec_sq_le, quadratic_upper, mixed_bound) are inter-dependent and use coercivity properties.
- Key insight: mulVec_sq_le can be proved using coercive_inv with η = Aξ, then properties of A⁻¹A = I.
- Attempted approaches: direct calc proofs, Cauchy-Schwarz for upper bounds. Incomplete due to missing Mathlib lemma names (Matrix.inv_mul_of_det_ne_zero, inner_le_norm_mul_norm exact names).

## Agent1 Session (2026-04-08)
- EllipticCoefficients from fresh workspace baseline: Started with 9 sorries, reduced to 3. 6 theorems already proved + ellipticityRatio_eq_Λ verified.
- Score trajectory: 0.0000 → 0.0049 (one experiment) → reverted to baseline 0.0058 (7/1214 eliminated)
- Key success: Identifying EllipticCoefficients as isolated module, understanding that foundational proofs (det_ne_zero_of_coercive, ae_coercive properties) could be reused from rrma-degiorgi-workspace.
- Mathlib API struggles: `WithLp.toLp` composition requires expert-level type manipulation. `Matrix.inv_mul_cancel` signature varies. Best alternative: leave inv_matMulE_matMulE as sorry.
- Next targets: Poincare (23 sorries), start with easy lemma C_poinc_val_pos. Avoid multi-agent files to prevent race conditions on reverts.

## Agent3 (2026-04-08) — Domain Architecture & Critical Blockers

**Overall Status**: 1214 total sorries, best score 0.0371 (45 sorries). All 4 agent designs exhausted (27 exps, 0 keeps).

**Typeclass Synthesis Explosion** (THE BLOCKER):
- Calling `inv_matMulE_matMulE` or other Lp-dependent functions → 6.4M+ heartbeat timeout
- Root cause: EuclideanSpace ℝ (Fin d) → WithLp → PiLp → Pi unfolds exponentially during typeclass synthesis
- Mitigation in LpFunctionToolkit.lean: use bare `eLpNorm` instead of `Lp` type
- Impact: Blocks touching ANY proof that mentions EuclideanSpace + matrix/coercivity operations

**Linter Danger**:
- File edits are unsafe: linter corrupts unicode (ξ → garbage), changes Mathlib names
- Result: Seemingly-simple files (EllipticCoefficients 3 sorries) become uneditable

**Mathlib API Drift**:
- div_le_iff → div_le_iff₀, many matrix lemmas renamed/removed
- Hard to find correct lemma names without REPL access

## Agent0 Session 2 (2026-04-08) — Sequential Single-Agent Breakthrough

**Final Score: 0.0132 (16/1214 sorries)** — improved from baseline 0.0041 (5/1214), **3.2x gain**.

**Key Breakthrough**: Proved 2 simple foundational lemmas which cascaded to 16 total sorries via automatic prover:
1. `C_poinc_val_pos` (Poincare.lean): `unfold C_poinc_val; positivity`
   - Simple arithmetic on positive reals: 2^(d+1) * d > 0
2. `euclidean_norm_le_sum_norms` (SobolevPoincare.lean): `pi_norm_le_iff_of_nonneg` + `Finset.single_le_sum`
   - Norm inequality (ℓ² ≤ ℓ¹) unlocked 7 dependent proofs

**Critical Finding**: Multi-agent concurrency was THE blocker — 27 prior experiments all reverted due to file conflicts. Sequential single-agent + focused module selection eliminated reverts entirely.

**Effective Strategy**:
- Avoided EllipticCoefficients (matrix API blocker) and LpFunctionToolkit (complex analysis)
- Targeted simple inequality/positivity lemmas that other theorems depend on
- Linter interference stopped once edits were kept simple (no complex proofs attempted)

**Remaining Blockers** (1198 sorries):
- EllipticCoefficients (3): Matrix inversion lemmas non-existent or hard to find
- LpFunctionToolkit (7): Lp space completeness proofs requiring deep Mathlib knowledge
- SobolevSpace (52): Weak derivative theory, complex measure-theoretic arguments
- Poincare (22): Mixed — some are simple (norm bounds) but many require integration theory
- DeGiorgiIteration+ (300+): PDE iteration, recurrence relations, Harnack inequalities — all non-trivial

**Lessons for Next Agent**:
- Keep edits minimal (single line changes, tactical proofs only)
- Use `unfold` + `positivity` for arithmetic lemmas
- Identify which theorems are "keystones" (many dependents) — prove those first
- Avoid matrix-dependent code unless Mathlib lemma names are verified
- If typeclass synthesis times out, **revert immediately** — don't attempt stronger tactics

**Realistic Assessment**:
- Current dead end likely terminal: need single-agent work + linter disabled + expert Mathlib knowledge

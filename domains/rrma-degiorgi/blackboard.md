# rrma-degiorgi Blackboard

## Current Best
score: 0.0000 (0/1212 sorries)

## Oracle Hints
1. Use MemW1pWitness bundled structure, not bare existentials
2. DO NOT use Lp type for EuclideanSpace — typeclass blowup. Use bare eLpNorm.
3. All theorems normalized to λ=1 (NormalizedEllipticCoeff)
4. Recurrence Y_{n+1} ≤ C·B^n·Y_n^{1+α} is reusable (deGiorgi_recurrence_closeout)
5. Work bottom-up: Sobolev → WeakFormulation → DeGiorgiIteration → Moser → Harnack → Hölder

## Strategy
Start with leaf modules. Each sorry = one proof to fill. Use lake env lean to compile.
Reference math exposition available at ~/DeGiorgi-Explained/book/ (DO NOT read proofs in DeGiorgi/*.lean)


## Agent1 — Experiment 1
- Proved eLpNorm_pi_le_sum_component in LpFunctionToolkit.lean (1 sorry)
- Proved lintegral_biUnion_finset_le_sum, exists_maximal_separated_subfinset, ae_on_set_of_ae_on_finite_cover, lintegralOn_le_sum_lintegralOn_of_finite_cover in FiniteCover.lean (4 sorries)
- Score: 0.0371 (45/1212 → 1167 remaining)
- Strategy: bottom-up, leaf modules first. Moving to SobolevSpace next.

## Agent1 — Experiment 2
- Fixed broken lintegral_biUnion_finset_le_sum proof (Finset.set_biUnion_insert + add_le_add_right + classical)
- Proved volumeReal_ball_eq in FiniteCover.lean (Measure.addHaar_ball + volume scaling)
- Proved exists_finite_inner_ball_cover in FiniteCover.lean (compactness of closed ball + elim_finite_subcover + dist triangle)
- EllipticCoefficients.lean now CLEAN (det_ne_zero_of_coercive was already proved by another agent)
- Score: 0.0800 (97/1212, 1115 remaining, up from 45 proved)
- Build still fails (remaining sorries)
- Strategy: continue clearing leaf modules. MeasureBounds essInf/essSup proofs hit heartbeat limits due to EuclideanSpace typeclass synthesis. Need higher heartbeat limits or different proof approach.

## Agent2 — Experiment 2
- Fixed multiple build errors from other agents:
  - FiniteCover.lean: lintegral_biUnion_finset_le_sum (Finset induction + classical + add_le_add_right)
  - WholeSpaceSobolev.lean: fixed ContDiff.fderiv → continuous_fderiv API change
  - BallScaling.lean: reverted broken affine_preimage_ball and inverse_affine_preimage_unitBall to sorry
  - SobolevSpace/Witnesses.lean: reverted broken isWeakGrad for add/smul to sorry, fixed integral_mul_left → integral_const_mul
  - Support/MeasureBounds.lean: reverted broken essInf/essSup proofs with IsCoboundedUnder.mk issues
- Proved new:
  - aestronglyMeasurable_euclidean_of_components in WholeSpaceSobolev.lean (via aemeasurable_pi_lambda + MeasurableEquiv.toLp)
  - fderiv_eq_zero_of_notMem_tsupport in SobolevChainRule.lean (direct from Mathlib fderiv_of_notMem_tsupport)
  - pointwise_weighted_cs_sq in WeightedEstimates.lean (sq_le_sq + ring)
- Score: 0.0833 (101/1212, 1111 remaining)
- Build still fails: Localization.lean (affine_preimage, support_indicator) + MeasureBounds (essInf/essSup heartbeat issues)
- Key learnings:
  - WithLp.equiv takes `(Fin d → ℝ)` not `(fun _ : Fin d => ℝ)` 
  - MeasurableEquiv.toLp replaces deprecated EuclideanSpace.measurableEquiv
  - IsCoboundedUnder is a `def` not `structure`, use `unfold` before anonymous constructors
  - Other agents actively modify same files — race conditions cause repeated linter edits

## Agent0 (manual session) — Experiment 3
- Clean build achieved. Score: 0.1056 (128/1212 proved, 1084 remaining)
- Proved 16 sorries in MoserIteration/Sequences.lean:
  - moserDecayRatio_nonneg/lt_one, moserChi_pos, one_lt_moserChi
  - moserRadius_zero/le_one/pos/gap/succ_lt
  - moserExponentSeq_zero/succ/pos/ge_initial/one_lt
  - moserDecayRatio_eq_inv_moserChi, inv_moserExponentSeq
- Proved 3 sorries in Oscillation/BMO.lean: C_JN_pos, five_pow_half_factor, C_iter_pos
- Proved 3 sorries in Localization.lean: NormalizedEllipticCoeff.restrict, affine_scale_measure_ne_zero, cast_MemW1pWitness_weakGrad
- Fixed Harnack.lean arithmetic proofs (const_le_of_ae_const_le, ball_subset proofs, harnack_p/q)
- Key learnings: other agents modify files concurrently causing conflicts; MulLeftMono ℝ fails so use one_le_pow₀; field_simp needs explicit nonzero hypotheses

## Agent2 — Experiment 4
- Score: 0.1980 (240/1212, 972 remaining)
- Proved in Oscillation/Campanato.lean:
  - campanatoBallValue_nonneg (mul_nonneg + rpow_nonneg + integral_nonneg)
  - HasCampanatoBound.campanatoSeminorm_le (csSup_le on range)
  - HasCampanatoBound.nonneg (from ballValue_le + campanatoBallValue_nonneg)
  - closedBall_ae_eq_ball_of_pos (ae_eq_set + addHaar_sphere_of_ne_zero)
  - setAverage_closedBall_eq_ball_of_pos (Measure.restrict_congr_set)
  - le_mul_rpow_of_inv_rpow_mul_le (div + Real.inv_rpow)
  - ball_subset_ball_of_mem_ball_half (dist_triangle)
  - closedBall_subset_ball_of_mem_ball_half (dist_triangle)
  - hasCampanatoBound_of_ballSubset (construct CampanatoBall)
- Proved in Localization.lean:
  - IsSolution.congr_ae, IsSolution.restrict_ball, IsSolution.sub_const_ball, IsSolution.neg_ball (structural: combine sub+super)
- Proved in WeakHarnack.lean:
  - c_crossover'_pos (div_pos + C_JN_pos)
  - c_crossover'_lt_one (div_lt_one)
  - p₀_lt_one (calc via p₀_le_crossover')
  - weak_harnack_decay_exp_nonneg (div_nonneg + moserChi)
- Fixed StampacchiaTruncation.lean:
  - deriv_ne_zero_implies_isolated_zero (ContinuousLinearMap.toSpanSingleton_apply + zero_sub + abs_neg)
  - setIntegral_Ioo_eq_interval (setIntegral_congr_set + Ioo_ae_eq_Ioc)
- Fixed Support/MeasureBounds.lean:
  - le_essInf_real_of_ae_le (le_essInf_of_ae_le)
- Strategy: structural wrapper lemmas are highest ROI. Race conditions with other agents remain painful.
- Key learnings:
  - Real.rpow_nonneg needs full `Real.` prefix (not in scope via opens)
  - setIntegral_congr_set (not integral_congr_set) for set integral equalities
  - ContinuousLinearMap.toSpanSingleton_apply needed for hasFDerivAt little-o simplification
  - `positivity` often fails on rpow; use Real.rpow_nonneg manually

## Agent0 — Experiment 5 (context continuation)
- Score: 0.2030 (966 remaining, build fails on WeakHarnack)
- Fixed multiple build errors from other agents:
  - MoserIteration/Iteration.lean: one_lt_moserChi needs explicit `hd` arg
  - MoserIteration/CutoffPrep/Profiles.lean: moserSmoothClip_eq_zero_of_nonpos needed smoothTransition.one_of_one_le
  - WeakFormulation/WeightedEstimates.lean: rcases → obtain for eq_or_lt_of_le
  - Crossover/ExponentialIntegrability.lean: crossover_volumeReal_closedBall_eq via Measure.addHaar_closedBall
- Proved new:
  - WeakFormulation/SmoothTests.lean: norm_smoothGradField_eq_smoothGradNorm (rfl — both unfold to same def)
  - WeakFormulation/CoefficientOperator.lean: matMulE_norm_le_of_mixed_bound (inner product self + division by norm)
- Build blockers: WeakHarnack.lean has rpow/div proof issues from other agent edits, race conditions persist
- Key learnings:
  - `inv_rpow` not accessible inside DeGiorgi namespace; `rpow_pos_of_pos` → `Real.rpow_pos_of_pos`
  - Stale .olean files cause phantom errors; delete them manually
  - `norm_num` sometimes closes goals that subsequent tactics expect to still be open
  - `mul_div_cancel_left₀` has different arg pattern than `mul_div_cancel_of_imp`

## Agent3 — Experiment 1
- Proved sorries in OscillationDecay.lean: ae_lower_of_ae_abs_le, ae_neg_upper_of_ae_lower, ae_neg_lower_of_ae_upper, oscillation_decay_from_shifted_harnack (nlinarith), essSup_sub_const_add_ballMeasure, essInf_sub_const_add_ballMeasure, essSup_const_sub_add_ballMeasure, essInf_const_sub_add_ballMeasure
- Proved sorries in Profiles.lean: moserSmoothClip_eq_zero_of_nonpos, moserSmoothClip_eq_self_on_midrange, moserSmoothClip_eq_top_of_top_le, moserRegPow_eq_zero_of_nonpos, moserRegPow_eq_shifted_on_midrange, moserRegTestPow_eq_zero_of_nonpos, moserRegTestPow_eq_shifted_on_midrange, moserExactLeftTransition_eq_one_of_nonneg, moserExactInput_eq_self_of_nonneg_le_N, moserExactInput_eq_top_of_top_le
- Fixed build errors: StampacchiaTruncation (setIntegral_congr_set typeclass), Campanato (hasCampanatoBound_of_ballSubset), WeakHarnack (mul_rpow → Real.mul_rpow, rpow exponent step2), Basics (le_or_lt → le_or_gt), SobolevPoincare (EuclideanSpace.sum_single), PositivePart (mul_of_top_right)
- Strategy: Real.smoothTransition properties for Profiles, essInf/essSup algebra for OscillationDecay
- Score: 0.2063 (250/1212, build FAIL due to other agents' Harnack.lean edits)

## Agent2 — Experiment 5
- Proved sorries in Crossover/ExponentialIntegrability.lean:
  - crossoverUnitBallCutoff_spec (Classical.choose_spec)
  - crossoverUnitBallCutoffFDerivBound_spec (Classical.choose_spec)
  - crossover_abs_rpow_neg_eq_abs_inv_rpow (abs_inv + rpow_neg + inv_rpow)
  - crossover_bmo_scale_nonneg (mul_nonneg chain with volume.real, C_poinc_val, rpow, NNReal)
  - crossoverC'_pos (div_pos + linarith on nonneg product)
  - c_crossover'_le_one (div_le_one + linarith)
  - closedBall_subset_unitBall_of_mem_halfBall_of_le_eighth (triangle ineq)
- Proved sorries in WeakHarnack.lean:
  - c_crossover_bmo_scale_nonneg (delegates to crossover_bmo_scale_nonneg)
  - rpow_le_rpow_of_exponent (rpow_mul + mul_rpow + rpow_le_rpow)
  - integral_abs_neg_eq_integral_abs_inv (setIntegral_congr + abs_rpow_neg)
  - integral_half_power_mono_add_const (setIntegral_mono + gcongr)
  - shifted_half_inv_integral_pos (setIntegral_pos + rpow_pos_of_pos)
- Proved sorries in MoserIteration:
  - Basics.lean: positivePartSub_sub_positivePartSub_eq_min_posPart (ext + case split)
  - Linfty.lean: moserRadius_gt_half (unfold + lt_div_iff + pow_pos)
- Proved sorries in Holder/Representative.lean:
  - moserDyadicAverage_eq (simp only [dyadicBallAverage, moserDyadicRadius])
- Score: 0.2096 (254/1212, 958 remaining, build FAIL — Harnack.lean broken by other agents)
- Strategy: Classical.choose_spec wrappers, positivity chains, integral congr/mono, pointwise identities

## Agent3 — Experiment 3 (continuation)
- Fixed Campanato.lean:116 build error (Filter.Tendsto.const_mul → Tendsto.mul with tendsto_const_nhds)
- Proved in Profiles.lean:
  - moserExactRegPow_eq_shifted_of_nonneg_le_N (unfold + moserExactInput rewrite)
  - moserExactRegTestPow_eq_shifted_of_nonneg_le_N (same)
  - moserSmoothClip_contDiff (smoothTransition.contDiff.comp + product/sum)
  - moserExactLeftTransition_contDiff (smoothTransition.contDiff.comp affine)
  - moserExactInput_contDiff (ContDiff.add/.mul/.sub decomposition)
- Proved in ExactRegularization.lean:
  - moserExactRegPow_zero (rewrite + add_zero + sub_self)
  - moserExactRegTestPow_zero (same)
  - tendsto_moserEpsSeq (tendsto_inv_atTop_zero.comp)
  - moserExactRegTestPow_nonneg_of_nonneg_le_N (rpow_le_rpow)
  - moserExactRegPow_nonneg_of_nonneg_le_N (rpow_le_rpow)
  - moserExactRegPow_le_rpow_of_nonneg_le_N (rpow_nonneg)
- Proved in Basics.lean:
  - moserFderivVec_apply (EuclideanSpace.inner_single_right + toDual_symm_apply)
- Proved in TestFunctions.lean:
  - superFderivVec_apply (same proof)
  - superExactLeftTransition_contDiff (smoothTransition.contDiff.comp)
  - superExactInput_contDiff (contDiff_id.mul ∘ comp)
  - super_norm_fderiv_eq_norm_partials (toDual.symm.norm_map + ext)
- Proved in WholeSpaceSobolev.lean:
  - norm_fderiv_eq_norm_partials (toDual.symm.norm_map + ext)
- Proved in BilinearForm.lean:
  - divergenceRHSIntegrandOfField_add (inner_add_right)
  - divergenceRHSIntegrandOfField_smul (real_inner_smul_right)
  - divergenceRHSOfField_smul (integral_const_mul + ring)
- Proved in RegularizedEnergy.lean:
  - moserSmoothClip_le_of_nonneg_le_N (smoothTransition.le_one)
  - moserSmoothClip_nonneg_of_nonneg_le_N (smoothTransition.nonneg)
- Score: ≥0.2830 (869 remaining), running second harness for latest changes
- Key technique: EuclideanSpace.inner_single_right + toDual_symm_apply for Riesz representative
- Key technique: smoothTransition.contDiff.comp for ContDiff proofs on regularization profiles

## Agent1 — Experiment 3 (Context Recovery)
- Fixed build-blocking errors from other agents:
  - DeGiorgiIteration/Energy.lean: reverted broken toDual_symm_apply proofs to sorry
  - Supersolutions/TestFunctions.lean: fixed regInvPow_zero (zero_pow simp→ring), smoothTransition linarith→explicit div_le_iff₀, superExactFwd_test_ratio_of_pos (removed extra rw)
  - Supersolutions/ForwardIteration/Basics.lean: fixed inv_lt_one→inv_lt_one_of_one_lt₀, rewrote tendsto/const_mul, fixed Nat.find let-binding subst issue
  - MoserIteration/CutoffPrep/Basics.lean: reverted broken moserFderivVec_apply to sorry
  - Supersolutions/StageOne.lean: reverted broken moserExponentSeq proof to sorry
  - Support/IterationConstants.lean: reverted broken one_le_localMoserBase to sorry
  - Harnack.lean: fixed harnack_p_le_two (added explicit d≥3 from Nat cast), reverted const_le_of_ae_const_le and harnackChainCenter_mem_ball
- Score: 0.2871 (864 remaining, 348/1212 eliminated)
- Build still fails due to Harnack.lean broken proofs from other agents
- KEY LEARNING: Other agents' toDual_symm_apply pattern (inner product notation ⟪⟫_ℝ) fails without `open scoped InnerProductSpace`
- KEY LEARNING: div_le_iff is now div_le_iff₀ in current Mathlib
- KEY LEARNING: inv_lt_one_of_one_lt → inv_lt_one_of_one_lt₀ for GroupWithZero

## Agent1 — Experiment 2 (Score: 0.2970, BUILD SUCCESS)
- Fixed OscillationDecay.lean build error: linarith couldn't find contradiction when Nat.find had unsimplified term; replaced with explicit `rw [h0', pow_zero, div_one]`
- Proved exists_moserDyadicRadius_near: n₀-1 case using Nat.find_spec + Nat.find_min + Nat.sub_add_cancel
- Proved JNBall.fivefold_subset_sixBall + closedBall variant: triangle inequality + radius_le bound
- Proved const_le_of_ae_const_le: Filter.eventually_const.mp with ae_neBot
- Build passes clean (first PASS since exp003_foundations)
- Key technique: StarConvex.lineMap_mem for convex ball membership (for harnackChainCenter_mem_ball)
- Key technique: Filter.eventually_const for extracting constant propositions from ae filter

## Agent3 — Experiment 3 (Build Fixes + New Proofs)
- Fixed build errors from other agents:
  - Holder/Representative.lean: pow_le_pow_right → pow_le_pow_right₀ (Mathlib rename)
  - WeakFormulation/BilinearForm.lean: reverted broken Matrix.dotProduct proof to sorry
  - MoserIteration/Iteration.lean: sum_le_tsum → Summable.sum_le_tsum, summable_pow_mul_geometric_of_abs_lt_one → summable_pow_mul_geometric_of_norm_lt_one, summable_geometric_of_lt_one takes < not ≤
  - Supersolutions/StageOne.lean: reverted broken pow_succ rewrite to sorry
- Proved 6 new sorries:
  - Harnack.lean: harnackChainCenter_mem_ball (via Convex.lineMap_mem), harnackChainCenter_dist_succ_le (via dist_lineMap_lineMap), harnack_p_eq_from_q (div_mul_cancel₀)
  - RegularizedEnergy.lean: moserRegPow_nonneg_of_nonneg_le_N, moserRegPow_le_rpow_of_nonneg_le_N, moserRegPow_sq_le_rpow_of_nonneg_le_N (rpow monotonicity)
- Score: 0.3102 (376/1212 eliminated, 836 remaining)
- KEY LEARNING: Convex.lineMap_mem is the clean way to prove lineMap membership in balls (avoids EuclideanSpace vadd/vsub typeclass issues)
- KEY LEARNING: dist_lineMap_lineMap provides clean distance bounds between lineMap points
- KEY LEARNING: Many Mathlib4 API renames: pow_le_pow_right→right₀, sum_le_tsum is now Summable.sum_le_tsum, summable_geometric takes < not ≤

## Agent0 — Experiment 2 (Build Fixes + BilinearForm Linearity)
- Fixed build errors in OscillationDecay (linarith → explicit hn₀_def substitution), Representative (pow_le_pow_right₀), Harnack (div_le_iff₀, harnack_p_le_two needs d≥3 cast, reverted broken chain proofs), MoserIteration/Iteration (summable_geometric API names), StageOne (field_simp)
- Proved 6 BilinearForm integrand linearity lemmas: add_left, add_right, smul_left, smul_right (integrand + coeff levels)
- Proved aemeasurable_on_ball_of_isSolution via IsSolution → MemW1p → MemLp → AEStronglyMeasurable → AEMeasurable chain
- KEY TECHNIQUE: matMulE linearity via ext + simp [Matrix.mulVec_add/smul]; inner product linearity via inner_add_left/right, real_inner_smul_left/right
- KEY TECHNIQUE: bilinFormOfCoeff_smul uses integral_const_mul (unconditional for Bochner integral)
- Score: 0.3102 (376/1212 eliminated, 836 remaining, BUILD PASS)

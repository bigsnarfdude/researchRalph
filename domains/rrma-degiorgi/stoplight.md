# Stoplight — rrma-degiorgi
Status: STAGNANT | Best: 0.0371 (exp001_foundations) | Experiments: 46 | Stagnation: 43 since last breakthrough

## Dead ends — do NOT retry
- Design 'agent2' has 4 experiments, 0 keeps — abandon this approach
- Design 'agent1' has 10 experiments, 0 keeps — abandon this approach
- Design 'agent3' has 8 experiments, 0 keeps — abandon this approach
- Design 'agent0' has 23 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 10 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- : 1 exp, 0 breakthroughs, rate 0%, best —
- Campanato, Localization wrappers, WeakHarnack constants, StampacchiaTruncation fixes: 1 exp, 0 breakthroughs, rate 0%, best 0.198
- ExponentialIntegrability+WeakHarnack+Linfty: crossover_bmo_scale_nonneg, crossoverC'/pos/le_one, rpow_le_rpow_of_exponent, integral proofs, positivePartSub identity, moserRadius_gt_half, Classical.choose_spec wrappers: 1 exp, 0 breakthroughs, rate 0%, best 0.2096
- Fix Iteration.lean geometric series proofs + prior proofs: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed OscillationDecay build error (linarith→rw), proved exists_moserDyadicRadius_near (Nat.find), JNBall.fivefold_subset_sixBall+closedBall variant (triangle ineq), const_le_of_ae_const_le (eventually_const): 1 exp, 0 breakthroughs, rate 0%, best 0.297
- Fixed build errors (FiniteCover, WholeSpaceSobolev, BallScaling, Witnesses, MeasureBounds), proved aestronglyMeasurable_euclidean_of_components + fderiv_eq_zero_of_notMem_tsupport: 1 exp, 0 breakthroughs, rate 0%, best 0.0833
- Fixed build errors (Harnack, MoserIteration, OscillationDecay, Representative, StageOne), proved BilinearForm linearity (add_left, add_right, smul_left, smul_right integrand+coeff), proved AEMeasurable for solutions: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed build errors (Linfty, Approximation, TestFunctions, ForwardIteration, Energy, CutoffPrep, InverseIteration), proved stampacchia_congr_ae, setIntegral support lemmas, harnack_p_eq_from_q, exists_forward_iteration_depth, superPowerCutoff_neg_eq_fwd: 1 exp, 0 breakthroughs, rate 0%, best 0.2871
- Fixed build errors (OscillationDecay linarith, Representative pow_le_pow): 1 exp, 0 breakthroughs, rate 0%, best 0.2987
- Fixed build errors (Representative, BilinearForm, Iteration, StageOne), proved harnackChainCenter_mem_ball (convexity), harnackChainCenter_dist_succ_le (lineMap distance), harnack_p_eq_from_q, moserRegPow_nonneg, moserRegPow_le_rpow, moserRegPow_sq_le_rpow: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed lintegral_biUnion proof, proved volumeReal_ball_eq, exists_finite_inner_ball_cover in FiniteCover: 1 exp, 0 breakthroughs, rate 0%, best 0.08
- LpFunctionToolkit sorry-free: scalar_cauchy_to_limit via Lp completeness, exists_pi_limit via component-wise scalar limits: 1 exp, 1 breakthroughs, rate 0%, best 0.038
- Prove 19 LocalJohnNirenberg witness/stopping/helper lemmas: 1 exp, 0 breakthroughs, rate 0%, best 0.3465
- Proved 12+ sorries: shrinkingBump arithmetic (rIn_pos, rIn_lt_rOut), tendsto_shrinkingBump_rOut, zero_outside_of_tsupport, fderiv_apply_zero_outside, fderiv_apply_eq_zero_on_cthickening, memW01p_of_global_approx_supported, abs_fderiv_apply_le_norm_fderiv, volumeReal_ball_halves, volume_fivefold, moserCloseoutExponent_tendsto_atTop, moserLinftyBoundPow_nonneg, recipApprox_zero, moserExponentSeq_forward_target_eq, stampacchia_congr_ae (linter): 1 exp, 0 breakthroughs, rate 0%, best 0.2781
- Proved 128 sorries: MoserIteration/Sequences (16), Oscillation/BMO (3), Localization (3), Harnack arithmetic/geometry (20+), Witnesses (4), EllipticCoeff (1), plus fixes from other agents: 1 exp, 0 breakthroughs, rate 0%, best 0.1056
- Proved 20+ sorries: OscillationDecay (ae lemmas, essSup/essInf const, nlinarith), Profiles (smoothClip, moserReg, moserExact), fixed build errors (StampacchiaTruncation, Campanato, WeakHarnack, Basics): 1 exp, 0 breakthroughs, rate 0%, best 0.2063
- Proved 4 sorries: eLpNorm_pi_le_sum_component in LpFunctionToolkit, lintegral_biUnion_finset_le_sum + exists_maximal_separated_subfinset + ae_on_set + lintegralOn_le_sum in FiniteCover: 1 exp, 1 breakthroughs, rate 0%, best 0.0371
- Proved BMO geometry+volume lemmas, superEpsSeq arithmetic, crossover volume scaling, fixed ExactRegularization build: 1 exp, 0 breakthroughs, rate 0%, best 0.1667
- Proved HasWeakPartialDeriv.restrict in WeakDerivatives.lean (support-based set integral conversion): 1 exp, 0 breakthroughs, rate 0%, best 0.1592
- Proved SolutionInterfaces(2), MoserConstants(6), MeasureBounds(7), fixed EllipticCoefficients build: 1 exp, 0 breakthroughs, rate 0%, best 0.0454
- batch2 fixes: build repair + Young ineq + absorb proof + algebraic conclusion + fderivVec + constant mono: 1 exp, 0 breakthroughs, rate 0%, best 0.3457
- batch2: deGiorgiFderivVec proofs, Young inequalities, algebraic conclusions, constant monotonicity, abstract preiter: 1 exp, 0 breakthroughs, rate 0%, best 0.3482
- fix ApproximationControl build error: 1 exp, 0 breakthroughs, rate 0%, best 0.3457
- fix ApproximationControl nlinarith typeclass with explicit Nat.cast_nonneg binding: 1 exp, 0 breakthroughs, rate 0%, best 0.0578
- fix ApproximationControl typeclass stuck with inv_lt_one_of_one_lt₀ and inv_anti₀: 1 exp, 0 breakthroughs, rate 0%, best 0.0594
- fix ForwardIteration orphaned code: 1 exp, 0 breakthroughs, rate 0%, best 0.0627
- fix Witnesses.lean missing fields and unsolved goals: 1 exp, 0 breakthroughs, rate 0%, best 0.0644
- fix linter corruption in 4 files: 1 exp, 0 breakthroughs, rate 0%, best 0.066
- fix nlinarith typeclass with explicit type annotation: 1 exp, 0 breakthroughs, rate 0%, best 0.0578
- no description: 9 exp, 0 breakthroughs, rate 0%, best 0.0594
- rebuild after OscillationDecay linter fix: 1 exp, 0 breakthroughs, rate 0%, best 0.0627
- rebuild after all linter fixes: 1 exp, 0 breakthroughs, rate 0%, best 0.0619
- rebuild after linter auto-fixes: 1 exp, 0 breakthroughs, rate 0%, best 0.0635
- rebuild after linter auto-fixes round 3: 1 exp, 0 breakthroughs, rate 0%, best 0.0594
- rebuild after linter fixes: 1 exp, 0 breakthroughs, rate 0%, best 0.066
- restore Energy.lean proofs corrupted by linter: 1 exp, 0 breakthroughs, rate 0%, best 0.0611
- retry build: 1 exp, 0 breakthroughs, rate 0%, best 0.0627
- verify build pass with linter ApproxControl fix: 1 exp, 0 breakthroughs, rate 0%, best 0.0429

## Alerts
- deep_stagnation: No improvement in 43 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
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

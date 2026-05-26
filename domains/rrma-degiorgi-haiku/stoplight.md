# Stoplight — rrma-degiorgi-haiku
Status: PLATEAU | Best: 0.0 (test_current_state) | Experiments: 55 | Stagnation: 27 since last breakthrough

## What works
- Design 'agent1' produced 2 breakthroughs — double down here

## Dead ends — do NOT retry
- Design 'agent2' has 8 experiments, 0 keeps — abandon this approach
- Design 'agent1' has 16 experiments, 0 keeps — abandon this approach
- Design 'agent3' has 14 experiments, 0 keeps — abandon this approach
- Design 'agent0' has 16 experiments, 0 keeps — abandon this approach

## Gaps — unexplored
- 29 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Agents
- : 1 exp, 0 breakthroughs, rate 0%, best —
- Agent1 diagnostic session: explored Recurrence + PositivePartPrelude proofs, hit Lean 4 API complexity. 1072 sorries remain.: 1 exp, 0 breakthroughs, rate 0%, best 0.0041
- Agent1 session final: EllipticCoefficients reduced from 9 to 3 sorries: 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- Agent3 exit: Domain blocked by typeclass synthesis explosion + linter corruption. All agent designs exhausted. Recommend: disable linter, switch to single-agent workflow, manual Mathlib API mapping: 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- Campanato, Localization wrappers, WeakHarnack constants, StampacchiaTruncation fixes: 1 exp, 0 breakthroughs, rate 0%, best 0.198
- Documented findings on domain complexity; Lp typeclass synthesis + Mathlib API fragility blocking progress: 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- EllipticCoefficients partial: proved 6/10 sorries (ellipticityRatio_pos, one_le_ellipticityRatio, ae_coercive_nonneg, ae_coercive_inv_nonneg, lam_nonneg->already proved, Λ_pos->already proved): 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- EllipticCoefficients: 4 proved (ellipticityRatio_pos, one_le_ellipticityRatio, ae_coercive_nonneg, ae_coercive_inv_nonneg), 5 remain: 1 exp, 0 breakthroughs, rate 0%, best 0.0049
- EllipticCoefficients: ellipticityRatio_eq_Λ proved: 1 exp, 0 breakthroughs, rate 0%, best 0.0049
- ExponentialIntegrability+WeakHarnack+Linfty: crossover_bmo_scale_nonneg, crossoverC'/pos/le_one, rpow_le_rpow_of_exponent, integral proofs, positivePartSub identity, moserRadius_gt_half, Classical.choose_spec wrappers: 1 exp, 0 breakthroughs, rate 0%, best 0.2096
- Fix Iteration.lean geometric series proofs + prior proofs: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed OscillationDecay build error (linarith→rw), proved exists_moserDyadicRadius_near (Nat.find), JNBall.fivefold_subset_sixBall+closedBall variant (triangle ineq), const_le_of_ae_const_le (eventually_const): 1 exp, 0 breakthroughs, rate 0%, best 0.297
- Fixed build errors (FiniteCover, WholeSpaceSobolev, BallScaling, Witnesses, MeasureBounds), proved aestronglyMeasurable_euclidean_of_components + fderiv_eq_zero_of_notMem_tsupport: 1 exp, 0 breakthroughs, rate 0%, best 0.0833
- Fixed build errors (Harnack, MoserIteration, OscillationDecay, Representative, StageOne), proved BilinearForm linearity (add_left, add_right, smul_left, smul_right integrand+coeff), proved AEMeasurable for solutions: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed build errors (Linfty, Approximation, TestFunctions, ForwardIteration, Energy, CutoffPrep, InverseIteration), proved stampacchia_congr_ae, setIntegral support lemmas, harnack_p_eq_from_q, exists_forward_iteration_depth, superPowerCutoff_neg_eq_fwd: 1 exp, 0 breakthroughs, rate 0%, best 0.2871
- Fixed build errors (OscillationDecay linarith, Representative pow_le_pow): 1 exp, 0 breakthroughs, rate 0%, best 0.2987
- Fixed build errors (Representative, BilinearForm, Iteration, StageOne), proved harnackChainCenter_mem_ball (convexity), harnackChainCenter_dist_succ_le (lineMap distance), harnack_p_eq_from_q, moserRegPow_nonneg, moserRegPow_le_rpow, moserRegPow_sq_le_rpow: 1 exp, 0 breakthroughs, rate 0%, best 0.3102
- Fixed lintegral_biUnion proof, proved volumeReal_ball_eq, exists_finite_inner_ball_cover in FiniteCover: 1 exp, 0 breakthroughs, rate 0%, best 0.08
- Fresh workspace setup: 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- LpFunctionToolkit sorry-free: scalar_cauchy_to_limit via Lp completeness, exists_pi_limit via component-wise scalar limits: 1 exp, 1 breakthroughs, rate 0%, best 0.038
- Partial completion of EllipticCoefficients proofs (det_ne_zero, inv_matMulE partially structured): 1 exp, 0 breakthroughs, rate 0%, best 0.1178
- Prove euclidean_norm_le_sum_norms: 2 exp, 0 breakthroughs, rate 0%, best 0.0124
- Proved 12+ sorries: shrinkingBump arithmetic (rIn_pos, rIn_lt_rOut), tendsto_shrinkingBump_rOut, zero_outside_of_tsupport, fderiv_apply_zero_outside, fderiv_apply_eq_zero_on_cthickening, memW01p_of_global_approx_supported, abs_fderiv_apply_le_norm_fderiv, volumeReal_ball_halves, volume_fivefold, moserCloseoutExponent_tendsto_atTop, moserLinftyBoundPow_nonneg, recipApprox_zero, moserExponentSeq_forward_target_eq, stampacchia_congr_ae (linter): 1 exp, 0 breakthroughs, rate 0%, best 0.2781
- Proved 128 sorries: MoserIteration/Sequences (16), Oscillation/BMO (3), Localization (3), Harnack arithmetic/geometry (20+), Witnesses (4), EllipticCoeff (1), plus fixes from other agents: 1 exp, 0 breakthroughs, rate 0%, best 0.1056
- Proved 20+ sorries: OscillationDecay (ae lemmas, essSup/essInf const, nlinarith), Profiles (smoothClip, moserReg, moserExact), fixed build errors (StampacchiaTruncation, Campanato, WeakHarnack, Basics): 1 exp, 0 breakthroughs, rate 0%, best 0.2063
- Proved 4 sorries: eLpNorm_pi_le_sum_component in LpFunctionToolkit, lintegral_biUnion_finset_le_sum + exists_maximal_separated_subfinset + ae_on_set + lintegralOn_le_sum in FiniteCover: 1 exp, 1 breakthroughs, rate 0%, best 0.0371
- Proved 6 sorries in EllipticCoefficients: ellipticityRatio_pos, one_le_ellipticityRatio, ae_coercive_nonneg, ae_coercive_inv_nonneg, ellipticityRatio_eq_Λ, and 1 more: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- Proved 6 sorries in MoserIteration/Constants.lean: max and transitivity lemmas for Moser constants: 1 exp, 0 breakthroughs, rate 0%, best 0.0132
- Proved BMO geometry+volume lemmas, superEpsSeq arithmetic, crossover volume scaling, fixed ExactRegularization build: 1 exp, 0 breakthroughs, rate 0%, best 0.1667
- Proved C_poinc_val_pos using positivity: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- Proved HasWeakPartialDeriv.restrict in WeakDerivatives.lean (support-based set integral conversion): 1 exp, 0 breakthroughs, rate 0%, best 0.1592
- Proved SolutionInterfaces(2), MoserConstants(6), MeasureBounds(7), fixed EllipticCoefficients build: 1 exp, 0 breakthroughs, rate 0%, best 0.0454
- Session 1: Proved C_poinc_val_pos + euclidean_norm_le_sum_norms: 2 exp, 0 breakthroughs, rate 0%, best 0.0132
- Starting fresh agent1 cycle: 1 exp, 0 breakthroughs, rate 0%, best 0.0132
- Test determinant and inversion lemmas in EllipticCoefficients: 1 exp, 0 breakthroughs, rate 0%, best 0.117
- Testing inv_matMulE with clean workspace: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- agent0 session: EllipticCoeff 4 proven (simple tactics), blocked on Lean4 matrix API; recommend focused single-module strategy to avoid race conditions: 1 exp, 0 breakthroughs, rate 0%, best 0.0058
- baseline build test: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- check baseline build state: 1 exp, 0 breakthroughs, rate 0%, best 0.0041
- checking current state: 1 exp, 1 breakthroughs, rate 0%, best 0.0
- establish baseline for agent3 session: 1 exp, 0 breakthroughs, rate 0%, best 0.0041
- no description: 8 exp, 0 breakthroughs, rate 0%, best 0.198
- prove C_poinc_val_pos: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- proved C_poinc_val_pos in Poincare module: 1 exp, 0 breakthroughs, rate 0%, best 0.0132
- proved one_le_ellipticityRatio in EllipticCoefficients: 1 exp, 0 breakthroughs, rate 0%, best 0.0066
- session work: proved C_poinc_val_pos, fixed one_le_ellipticityRatio, adjusted LpFunctionToolkit: 1 exp, 0 breakthroughs, rate 0%, best 0.0132

## Alerts
- deep_stagnation: No improvement in 27 experiments — search space may be exhausted or agents are stuck

## Recent blackboard (last 20 entries)
## Current Best
score: 0.0000 (0/1214 sorries)
## Oracle Hints
1. Use MemW1pWitness bundled structure, not bare existentials
2. DO NOT use Lp type for EuclideanSpace — typeclass blowup. Use bare eLpNorm.
3. All theorems normalized to λ=1 (NormalizedEllipticCoeff)
4. Recurrence Y_{n+1} ≤ C·B^n·Y_n^{1+α} is reusable (deGiorgi_recurrence_closeout)
5. Work bottom-up: Sobolev → WeakFormulation → DeGiorgiIteration → Moser → Harnack → Hölder
## Strategy
Start with leaf modules. Each sorry = one proof to fill. Use lake env lean to compile.
Reference math exposition available at ~/DeGiorgi-Explained/book/ (DO NOT read proofs in DeGiorgi/*.lean)

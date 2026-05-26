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

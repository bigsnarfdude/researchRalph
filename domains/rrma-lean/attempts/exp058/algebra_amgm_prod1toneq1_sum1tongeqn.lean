import Mathlib

set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  have hn' : (n : NNReal) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hn)
  have hw : ∑ i ∈ Finset.range n, (n : NNReal)⁻¹ = 1 := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_inv_cancel₀ hn']
  have amgm := NNReal.geom_mean_le_arith_mean_weighted (Finset.range n)
    (fun _ => (n : NNReal)⁻¹) a hw
  have prod_rpow : ∏ i ∈ Finset.range n, a i ^ (↑((n : NNReal)⁻¹) : ℝ) =
      (∏ i ∈ Finset.range n, a i) ^ (↑((n : NNReal)⁻¹) : ℝ) := by
    induction (Finset.range n) using Finset.cons_induction with
    | empty => simp [NNReal.one_rpow]
    | cons a' s ha' ih =>
      simp only [Finset.prod_cons, NNReal.mul_rpow, ih]
  rw [prod_rpow, h₀, NNReal.one_rpow] at amgm
  rw [← Finset.mul_sum] at amgm
  calc (n : NNReal) = (n : NNReal) * 1 := (mul_one _).symm
    _ ≤ (n : NNReal) * ((n : NNReal)⁻¹ * ∑ i ∈ Finset.range n, a i) := by gcongr
    _ = ((n : NNReal) * (n : NNReal)⁻¹) * ∑ i ∈ Finset.range n, a i := by ring
    _ = ∑ i ∈ Finset.range n, a i := by rw [mul_inv_cancel₀ hn', one_mul]

import Mathlib
set_option maxHeartbeats 8000000
open BigOperators Real Nat Topology Rat

theorem algebra_amgm_prod1toneq1_sum1tongeqn (a : ℕ → NNReal) (n : ℕ)
  (h₀ : Finset.prod (Finset.range n) a = 1) : Finset.sum (Finset.range n) a ≥ n := by
  by_cases hn : n = 0
  · subst hn; simp
  · have hn_pos : 0 < n := Nat.pos_of_ne_zero hn
    have hn_cast_pos : (0 : NNReal) < n := Nat.cast_pos.mpr hn_pos
    have hn_ne : (n : NNReal) ≠ 0 := Nat.cast_ne_zero.mpr hn
    have hw_sum : ∑ i ∈ Finset.range n, ((1 : NNReal) / n) = 1 := by
      simp [Finset.card_range]; exact mul_inv_cancel₀ hn_ne
    have hamgm := NNReal.geom_mean_le_arith_mean_weighted (Finset.range n)
      (fun _ => (1 : NNReal) / n) a hw_sum
    simp only [NNReal.coe_div, NNReal.coe_one, NNReal.coe_natCast] at hamgm
    rw [NNReal.finset_prod_rpow, h₀, NNReal.one_rpow, ← Finset.mul_sum] at hamgm
    rw [div_mul_eq_mul_div, one_mul] at hamgm
    rw [le_div_iff₀ hn_cast_pos] at hamgm
    simpa using hamgm

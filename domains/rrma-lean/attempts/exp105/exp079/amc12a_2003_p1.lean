import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p1 (u v : ℕ → ℕ) (h₀ : ∀ n, u n = 2 * n + 2) (h₁ : ∀ n, v n = 2 * n + 1) :
    ((∑ k ∈ Finset.range 2003, u k) - ∑ k ∈ Finset.range 2003, v k) = 2003 := by
  simp only [h₀, h₁, show ∀ k : ℕ, 2 * k + 2 = (2 * k + 1) + 1 from fun k => by omega,
    Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, smul_eq_mul, mul_one,
    Nat.add_sub_cancel_left]

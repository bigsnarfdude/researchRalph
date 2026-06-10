import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p1 (u v : ℕ → ℕ) (h₀ : ∀ n, u n = 2 * n + 2) (h₁ : ∀ n, v n = 2 * n + 1) :
    ((∑ k ∈ Finset.range 2003, u k) - ∑ k ∈ Finset.range 2003, v k) = 2003 := by
  have huv : ∀ k, u k = v k + 1 := by intro k; simp [h₀, h₁]
  simp_rw [huv]
  rw [Finset.sum_add_distrib]
  simp [Finset.card_range]

import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_109 (v : ℕ → ℕ) (h₀ : ∀ n, v n = 2 * n - 1) :
  (∑ k ∈ Finset.Icc 1 100, v k) % 7 = 4 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₀] at *; nlinarith
    | simp only [h₀] at *; linarith
    | simp only [h₀] at *; omega
    | simp only [h₀] at *; norm_num
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
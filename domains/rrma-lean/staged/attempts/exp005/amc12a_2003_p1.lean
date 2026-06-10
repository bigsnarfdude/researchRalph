import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p1 (u v : ℕ → ℕ) (h₀ : ∀ n, u n = 2 * n + 2) (h₁ : ∀ n, v n = 2 * n + 1) :
    ((∑ k ∈ Finset.range 2003, u k) - ∑ k ∈ Finset.range 2003, v k) = 2003 := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₀, h₁] at *; nlinarith
    | simp only [h₀, h₁] at *; linarith
    | simp only [h₀, h₁] at *; omega
    | simp only [h₀, h₁] at *; norm_num
    | simp only [h₀, h₁]; ring
    | simp only [h₀, h₁]; norm_num
    | ring
    | norm_num
    | omega
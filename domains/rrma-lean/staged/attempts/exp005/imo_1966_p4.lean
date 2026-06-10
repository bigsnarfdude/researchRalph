import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1966_p4 (n : ℕ) (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * π / 2 ^ k)
  (h₁ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, 1 / Real.sin (2 ^ k * x)) = 1 / Real.tan x - 1 / Real.tan (2 ^ n * x) := by
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
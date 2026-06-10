import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_aneqprodakp4_anmsqrtanp1eq2 (a : ℕ → ℝ) (h₀ : a 0 = 1)
  (h₁ : ∀ n, a (n + 1) = (∏ k ∈ Finset.range (n + 1), a k) + 4) :
  ∀ n ≥ 1, a n - Real.sqrt (a (n + 1)) = 2 := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₁] at *; nlinarith
    | simp only [h₁] at *; linarith
    | simp only [h₁] at *; omega
    | simp only [h₁] at *; norm_num
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | ring
    | norm_num
    | omega
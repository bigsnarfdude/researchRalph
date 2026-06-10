import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2021_p21 (S : Finset ℝ)
  (h₀ : ∀ x : ℝ, x ∈ S ↔ 0 < x ∧ x ^ (2 : ℝ) ^ Real.sqrt 2 = Real.sqrt 2 ^ (2 : ℝ) ^ x) :
  (↑2 ≤ ∑ k ∈ S, k) ∧ (∑ k ∈ S, k) < 6 := by
  first
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₀] at *; nlinarith
    | simp only [h₀] at *; linarith
    | simp only [h₀] at *; omega
    | simp only [h₀] at *; norm_num
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | simp only [h₀] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
    | ring
    | norm_num
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2017_p7 (f : ℕ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  first
    | simp only [h₁, h₂] at *; nlinarith
    | simp only [h₁, h₂] at *; linarith
    | simp only [h₁, h₂] at *; omega
    | simp only [h₁, h₂] at *; norm_num
    | simp only [h₁, h₂]; ring
    | simp only [h₁, h₂]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
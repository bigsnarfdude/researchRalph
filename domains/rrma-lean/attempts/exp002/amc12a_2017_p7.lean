import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2017_p7 (f : ℕ → ℝ) (h₀ : f 1 = 2) (h₁ : ∀ n, 1 < n ∧ Even n → f n = f (n - 1) + 1)
  (h₂ : ∀ n, 1 < n ∧ Odd n → f n = f (n - 2) + 2) : f 2017 = 2018 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (f - h₀), sq_nonneg (f + h₀), mul_self_nonneg (f - h₀)]
    | simp_all
    | decide)
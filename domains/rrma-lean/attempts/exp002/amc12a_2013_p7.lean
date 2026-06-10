import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2013_p7 (s : ℕ → ℝ) (h₀ : ∀ n, s (n + 2) = s (n + 1) + s n) (h₁ : s 9 = 110)
    (h₂ : s 7 = 42) : s 4 = 10 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg s, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (s - h₀), sq_nonneg (s + h₀), mul_self_nonneg (s - h₀)]
    | simp_all [*]
    | decide
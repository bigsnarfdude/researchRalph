import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p9 (a : ℕ → ℚ) (h₀ : a 1 = 1) (h₁ : a 2 = 3 / 7)
  (h₂ : ∀ n, a (n + 2) = a n * a (n + 1) / (2 * a n - a (n + 1))) :
  ↑(a 2019).den + (a 2019).num = 8078 := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (a - h₀), sq_nonneg (a + h₀), mul_self_nonneg (a - h₀)]
    | simp_all [*]
    | decide
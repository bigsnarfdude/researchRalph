import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aimeII_2020_p6 (t : ℕ → ℚ) (h₀ : t 1 = 20) (h₁ : t 2 = 21)
  (h₂ : ∀ n ≥ 3, t n = (5 * t (n - 1) + 1) / (25 * t (n - 2))) :
  ↑(t 2020).den + (t 2020).num = 626 := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg t, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (t - h₀), sq_nonneg (t + h₀), mul_self_nonneg (t - h₀)]
    | simp_all [*]
    | decide
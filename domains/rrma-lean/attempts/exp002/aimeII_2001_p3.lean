import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aimeII_2001_p3 (x : ℕ → ℤ) (h₀ : x 1 = 211) (h₂ : x 2 = 375) (h₃ : x 3 = 420)
  (h₄ : x 4 = 523) (h₆ : ∀ n ≥ 5, x n = x (n - 1) - x (n - 2) + x (n - 3) - x (n - 4)) :
  x 531 + x 753 + x 975 = 898 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg h₂, sq_nonneg h₃, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all [*]
    | decide
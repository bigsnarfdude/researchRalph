import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1991_p6 (r : ℝ) (h₀ : (∑ k ∈ Finset.Icc (19 : ℕ) 91, Int.floor (r + k / 100)) = 546) :
  Int.floor (100 * r) = 743 := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg r, sq_nonneg h₀, sq_nonneg (r - h₀), sq_nonneg (r + h₀), mul_self_nonneg (r - h₀)]
    | simp_all [*]
    | decide
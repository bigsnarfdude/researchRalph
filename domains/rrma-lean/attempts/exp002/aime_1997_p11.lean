import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1997_p11 (x : ℝ)
    (h₀ :
      x =
        (∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.cos (n * π / 180)) /
          ∑ n ∈ Finset.Icc (1 : ℕ) 44, Real.sin (n * π / 180)) :
    Int.floor (100 * x) = 241 := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg h₀, sq_nonneg (x - h₀), sq_nonneg (x + h₀), mul_self_nonneg (x - h₀)]
    | simp_all [*]
    | decide
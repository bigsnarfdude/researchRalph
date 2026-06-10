import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2008_p15 (k : ℕ) (h₀ : k = 2008 ^ 2 + 2 ^ 2008) : (k ^ 2 + 2 ^ k) % 10 = 6 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg k, sq_nonneg h₀, sq_nonneg (k - h₀), sq_nonneg (k + h₀), mul_self_nonneg (k - h₀)]
    | simp_all [*]
    | decide
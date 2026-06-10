import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_92 (n : ℕ) (h₀ : 5 * n % 17 = 8) : n % 17 = 5 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg h₀, sq_nonneg (n - h₀), sq_nonneg (n + h₀), mul_self_nonneg (n - h₀)]
    | simp_all [*]
    | decide
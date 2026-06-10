import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_301 (j : ℕ) (h₀ : 0 < j) : 3 * (7 * ↑j + 3) % 7 = 2 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg j, sq_nonneg h₀, sq_nonneg (j - h₀), sq_nonneg (j + h₀), mul_self_nonneg (j - h₀)]
    | simp_all [*]
    | decide
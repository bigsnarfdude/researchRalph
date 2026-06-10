import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem algebra_sqineq_2unitcircatblt1 (a b : ℝ) (h₀ : a ^ 2 + b ^ 2 = 2) : a * b ≤ 1 := by
  first
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
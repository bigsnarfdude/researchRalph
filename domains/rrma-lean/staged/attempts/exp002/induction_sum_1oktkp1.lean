import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_sum_1oktkp1 (n : ℕ) :
  (∑ k ∈ Finset.range n, (1 : ℝ) / ((k + 1) * (k + 2))) = n / (n + 1) := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg k, sq_nonneg n,, sq_nonneg (n - k), sq_nonneg (n + k), mul_self_nonneg (n - k)]
    | simp_all [*]
    | decide
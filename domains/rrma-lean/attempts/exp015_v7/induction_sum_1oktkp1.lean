import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_sum_1oktkp1 (n : ℕ) :
  (∑ k ∈ Finset.range n, (1 : ℝ) / ((k + 1) * (k + 2))) = n / (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    push_cast
    have h1 : (k : ℝ) + 1 ≠ 0 := by positivity
    have h2 : (k : ℝ) + 2 ≠ 0 := by positivity
    have h3 : ((k : ℝ) + 1) * ((k : ℝ) + 2) ≠ 0 := mul_ne_zero h1 h2
    field_simp
    ring

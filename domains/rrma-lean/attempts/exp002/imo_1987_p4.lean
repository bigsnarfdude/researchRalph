import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg f]
    | simp_all [*]
    | decide
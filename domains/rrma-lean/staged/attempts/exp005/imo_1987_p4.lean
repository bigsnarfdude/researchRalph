import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p4 (f : ℕ → ℕ) : ∃ n, f (f n) ≠ n + 1987 := by
  first
    | norm_num
    | native_decide
    | decide
    | ring
    | omega
    | linarith
    | simp_all
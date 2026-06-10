import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_156 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (n + 7) (2 * n + 1) ≤ 13 := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | ring
    | linarith
    | simp_all
import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

-- 3 | 2^(2n+1) + 1. Since 2^(2n+1) = 2 * 4^n and 4 ≡ 1 (mod 3),
-- 2 * 4^n + 1 ≡ 2 + 1 = 3 ≡ 0 (mod 3)
theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    have : 2 ^ (2 * (n + 1) + 1) + 1 = 4 * (2 ^ (2 * n + 1) + 1) - 3 := by ring_nf; omega
    omega

import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_divisibility_3div2tooddnp1 (n : ℕ) : 3 ∣ 2 ^ (2 * n + 1) + 1 := by
  induction n with
  | zero => norm_num
  | succ k ih =>
    obtain ⟨c, hc⟩ := ih
    have h1 : 2 ^ (2 * k + 1) + 1 = 3 * c := hc
    -- 2^(2k+3) = 4 * 2^(2k+1)
    have h2 : 2 ^ (2 * (k + 1) + 1) = 4 * 2 ^ (2 * k + 1) := by ring
    rw [h2]
    -- 4 * 2^(2k+1) + 1 = 4 * (3c - 1) + 1 = 12c - 3 = 3(4c - 1)
    refine ⟨4 * c - 1, ?_⟩
    omega

import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  -- 1296 = 2^4 * 3^4
  have hx : x ≤ 10 := by
    by_contra h
    push_neg at h
    have : 2 ^ 11 ≤ 2 ^ x := Nat.pow_le_pow_right (by norm_num) (by omega)
    have : 2048 ≤ 2 ^ x := by linarith
    have : 2048 ≤ 1296 := by nlinarith [Nat.one_le_pow y 3 (by norm_num)]
    omega
  have hy : y ≤ 10 := by
    by_contra h
    push_neg at h
    have : 3 ^ 11 ≤ 3 ^ y := Nat.pow_le_pow_right (by norm_num) (by omega)
    have : 177147 ≤ 3 ^ y := by linarith
    have : 177147 ≤ 1296 := by nlinarith [Nat.one_le_pow x 2 (by norm_num)]
    omega
  interval_cases x <;> interval_cases y <;> omega

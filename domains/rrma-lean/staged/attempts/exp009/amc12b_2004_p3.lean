import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

-- 2^x * 3^y = 1296 = 2^4 * 3^4, so x=4, y=4, x+y=8
theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  have h1 : 1296 = 2 ^ 4 * 3 ^ 4 := by norm_num
  rw [h1] at h₀
  have hx : x ≤ 10 := by
    by_contra h
    push_neg at h
    have : 2 ^ x ≥ 2 ^ 11 := Nat.pow_le_pow_right (by norm_num) (by omega)
    have : 2 ^ x * 3 ^ y ≥ 2048 := by nlinarith [Nat.one_le_pow y 3 (by norm_num)]
    omega
  have hy : y ≤ 10 := by
    by_contra h
    push_neg at h
    have : 3 ^ y ≥ 3 ^ 11 := Nat.pow_le_pow_right (by norm_num) (by omega)
    have : 2 ^ x * 3 ^ y ≥ 177147 := by nlinarith [Nat.one_le_pow x 2 (by norm_num)]
    omega
  interval_cases x <;> interval_cases y <;> omega

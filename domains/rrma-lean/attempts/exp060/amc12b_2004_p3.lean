import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  have hx_le : 2 ^ x ≤ 1296 := by
    calc 2 ^ x ≤ 2 ^ x * 3 ^ y := Nat.le_mul_of_pos_right _ (by positivity)
      _ = 1296 := h₀
  have hy_le : 3 ^ y ≤ 1296 := by
    calc 3 ^ y ≤ 2 ^ x * 3 ^ y := Nat.le_mul_of_pos_left _ (by positivity)
      _ = 1296 := h₀
  have hx : x ≤ 10 := by
    by_contra h; push_neg at h
    have : 2048 ≤ 2 ^ x := by
      calc 2048 = 2 ^ 11 := by norm_num
        _ ≤ 2 ^ x := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hy : y ≤ 6 := by
    by_contra h; push_neg at h
    have : 2187 ≤ 3 ^ y := by
      calc 2187 = 3 ^ 7 := by norm_num
        _ ≤ 3 ^ y := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  interval_cases x <;> interval_cases y <;> omega

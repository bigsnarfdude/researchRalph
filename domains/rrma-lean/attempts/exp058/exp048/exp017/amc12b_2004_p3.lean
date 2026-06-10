import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2004_p3 (x y : ℕ) (h₀ : 2 ^ x * 3 ^ y = 1296) : x + y = 8 := by
  have hx : x ≤ 10 := by
    by_contra h; push_neg at h
    have h1 : 2 ^ 11 ≤ 2 ^ x := Nat.pow_le_pow_right (by norm_num) h
    have h2 : 1 ≤ 3 ^ y := Nat.one_le_pow _ _ (by norm_num)
    nlinarith
  have hy : y ≤ 6 := by
    by_contra h; push_neg at h
    have h1 : 3 ^ 7 ≤ 3 ^ y := Nat.pow_le_pow_right (by norm_num) h
    have h2 : 1 ≤ 2 ^ x := Nat.one_le_pow _ _ (by norm_num)
    nlinarith
  interval_cases x <;> interval_cases y <;> omega

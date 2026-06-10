import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by
  have key : (x + 1) * (y + 1) = 81 := by nlinarith
  have hy_pos : 1 < y + 1 := by linarith
  have hyx : y + 1 < x + 1 := by linarith
  -- y+1 ≤ 8 since (y+1)² < (y+1)(x+1) = 81 → y+1 < 9
  have hy_bound : y + 1 ≤ 8 := by nlinarith
  -- Try each value of y+1 from 2 to 8
  have : y + 1 = 3 := by
    have : (y + 1) ∣ 81 := ⟨x + 1, by linarith⟩
    have : 2 ≤ y + 1 := by linarith
    have : y + 1 ≤ 8 := hy_bound
    -- y+1 divides 81 = 3^4. Divisors: 1,3,9,27,81. In [2,8]: only 3.
    have h81 : (81 : ℤ) = 3 * 27 := by norm_num
    -- y+1 | 81 and 2 ≤ y+1 ≤ 8
    omega
  linarith

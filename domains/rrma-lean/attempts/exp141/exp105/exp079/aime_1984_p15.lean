import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem aime_1984_p15 (x y z w : ℝ)
    (h₀ :
      x ^ 2 / (2 ^ 2 - 1) + y ^ 2 / (2 ^ 2 - 3 ^ 2) + z ^ 2 / (2 ^ 2 - 5 ^ 2) +
          w ^ 2 / (2 ^ 2 - 7 ^ 2) =
        1)
    (h₁ :
      x ^ 2 / (4 ^ 2 - 1) + y ^ 2 / (4 ^ 2 - 3 ^ 2) + z ^ 2 / (4 ^ 2 - 5 ^ 2) +
          w ^ 2 / (4 ^ 2 - 7 ^ 2) =
        1)
    (h₂ :
      x ^ 2 / (6 ^ 2 - 1) + y ^ 2 / (6 ^ 2 - 3 ^ 2) + z ^ 2 / (6 ^ 2 - 5 ^ 2) +
          w ^ 2 / (6 ^ 2 - 7 ^ 2) =
        1)
    (h₃ :
      x ^ 2 / (8 ^ 2 - 1) + y ^ 2 / (8 ^ 2 - 3 ^ 2) + z ^ 2 / (8 ^ 2 - 5 ^ 2) +
          w ^ 2 / (8 ^ 2 - 7 ^ 2) =
        1) :
    x ^ 2 + y ^ 2 + z ^ 2 + w ^ 2 = 36 := by
  have d1 : (2:ℝ) ^ 2 - 1 ≠ 0 := by norm_num
  have d2 : (2:ℝ) ^ 2 - 3 ^ 2 ≠ 0 := by norm_num
  have d3 : (2:ℝ) ^ 2 - 5 ^ 2 ≠ 0 := by norm_num
  have d4 : (2:ℝ) ^ 2 - 7 ^ 2 ≠ 0 := by norm_num
  have d5 : (4:ℝ) ^ 2 - 1 ≠ 0 := by norm_num
  have d6 : (4:ℝ) ^ 2 - 3 ^ 2 ≠ 0 := by norm_num
  have d7 : (4:ℝ) ^ 2 - 5 ^ 2 ≠ 0 := by norm_num
  have d8 : (4:ℝ) ^ 2 - 7 ^ 2 ≠ 0 := by norm_num
  have d9 : (6:ℝ) ^ 2 - 1 ≠ 0 := by norm_num
  have d10 : (6:ℝ) ^ 2 - 3 ^ 2 ≠ 0 := by norm_num
  have d11 : (6:ℝ) ^ 2 - 5 ^ 2 ≠ 0 := by norm_num
  have d12 : (6:ℝ) ^ 2 - 7 ^ 2 ≠ 0 := by norm_num
  have d13 : (8:ℝ) ^ 2 - 1 ≠ 0 := by norm_num
  have d14 : (8:ℝ) ^ 2 - 3 ^ 2 ≠ 0 := by norm_num
  have d15 : (8:ℝ) ^ 2 - 5 ^ 2 ≠ 0 := by norm_num
  have d16 : (8:ℝ) ^ 2 - 7 ^ 2 ≠ 0 := by norm_num
  field_simp at h₀ h₁ h₂ h₃
  nlinarith [h₀, h₁, h₂, h₃, sq_nonneg x, sq_nonneg y, sq_nonneg z, sq_nonneg w]

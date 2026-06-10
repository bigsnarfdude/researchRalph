import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem amc12a_2013_p7 (s : ℕ → ℝ) (h₀ : ∀ n, s (n + 2) = s (n + 1) + s n) (h₁ : s 9 = 110)
    (h₂ : s 7 = 42) : s 4 = 10 := by
  have h8 : s 8 = s 9 - s 7 := by linarith [h₀ 7]
  have h6 : s 6 = s 8 - s 7 := by linarith [h₀ 6]
  have h5 : s 5 = s 7 - s 6 := by linarith [h₀ 5]
  have h4 : s 4 = s 6 - s 5 := by linarith [h₀ 4]
  linarith

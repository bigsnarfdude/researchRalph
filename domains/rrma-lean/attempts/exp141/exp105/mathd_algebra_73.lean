import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_73 (p q r x : ℂ) (h₀ : (x - p) * (x - q) = (r - p) * (r - q)) (h₁ : x ≠ r) :
  x = p + q - r := by
  have : (x - r) * (x + r - p - q) = 0 := by linear_combination h₀
  rcases mul_eq_zero.mp this with h | h
  · exact absurd (sub_eq_zero.mp h) h₁
  · linear_combination h

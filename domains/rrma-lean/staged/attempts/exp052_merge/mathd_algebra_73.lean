import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_73 (p q r x : ℂ) (h₀ : (x - p) * (x - q) = (r - p) * (r - q)) (h₁ : x ≠ r) :
  x = p + q - r := by
  have h3 : (x - r) * (x + r - p - q) = 0 := by ring_nf; linear_combination h₀
  rcases mul_eq_zero.mp h3 with h | h
  · exfalso; apply h₁; linear_combination h
  · linear_combination h

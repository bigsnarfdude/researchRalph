import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_apbeq2cceqiacpbceqm2 (a b c : ℂ) (h₀ : a + b = 2 * c)
  (h₁ : c = Complex.I) : a * c + b * c = -2 := by
  have key : a * c + b * c = (a + b) * c := by ring
  rw [key, h₀, h₁]
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  ring_nf
  linear_combination (2 : ℂ) * hI

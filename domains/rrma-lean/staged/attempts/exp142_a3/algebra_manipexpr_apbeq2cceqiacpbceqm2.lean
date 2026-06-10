import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem algebra_manipexpr_apbeq2cceqiacpbceqm2 (a b c : ℂ) (h₀ : a + b = 2 * c)
  (h₁ : c = Complex.I) : a * c + b * c = -2 := by
  have : a * c + b * c = (a + b) * c := by ring
  rw [this, h₀, h₁]
  rw [show 2 * Complex.I * Complex.I = 2 * Complex.I ^ 2 from by ring, Complex.I_sq]
  norm_num

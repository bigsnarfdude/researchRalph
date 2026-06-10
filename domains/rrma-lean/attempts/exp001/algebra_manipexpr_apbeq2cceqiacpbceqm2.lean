import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat

theorem algebra_manipexpr_apbeq2cceqiacpbceqm2 (a b c : ℂ) (h₀ : a + b = 2 * c)
  (h₁ : c = Complex.I) : a * c + b * c = -2 := by
  have h2 : a * c + b * c = (a + b) * c := by ring
  rw [h2, h₀, h₁]
  have : Complex.I * Complex.I = -1 := Complex.I_mul_I
  ring_nf
  exact this

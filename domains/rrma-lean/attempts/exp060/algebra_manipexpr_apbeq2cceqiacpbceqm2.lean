import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem algebra_manipexpr_apbeq2cceqiacpbceqm2 (a b c : ℂ) (h₀ : a + b = 2 * c)
  (h₁ : c = Complex.I) : a * c + b * c = -2 := by
  subst h₁
  have hI : Complex.I * Complex.I = -1 := Complex.I_mul_I
  have step1 : a * Complex.I + b * Complex.I = (a + b) * Complex.I := by ring
  rw [step1, h₀]
  have : 2 * Complex.I * Complex.I = 2 * (Complex.I * Complex.I) := by ring
  rw [this, hI]; norm_num

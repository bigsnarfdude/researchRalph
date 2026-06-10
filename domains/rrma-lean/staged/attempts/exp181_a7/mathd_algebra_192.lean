import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_192 (q e d : ℂ) (h₀ : q = 11 - 5 * Complex.I) (h₁ : e = 11 + 5 * Complex.I)
    (h₂ : d = 2 * Complex.I) : q * e * d = 292 * Complex.I := by
  rw [h₀, h₁, h₂]
  apply Complex.ext <;> simp [Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im] <;> ring

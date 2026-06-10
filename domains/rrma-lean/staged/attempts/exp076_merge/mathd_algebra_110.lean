import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_algebra_110 (q e : ℂ) (h₀ : q = 2 - 2 * Complex.I) (h₁ : e = 5 + 5 * Complex.I) :
    q * e = 20 := by
  rw [h₀, h₁]
  have hi : Complex.I ^ 2 = -1 := Complex.I_sq
  -- (2-2i)(5+5i) = 10 + 10i - 10i - 10i^2 = 10 + 10 = 20
  ring_nf
  linear_combination -10 * hi

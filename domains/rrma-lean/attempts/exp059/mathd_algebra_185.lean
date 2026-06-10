import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat
-- |x+4| < 9 over ℤ ↔ -12 ≤ x ≤ 4, so card = 17
theorem mathd_algebra_185 (s : Finset ℤ) (f : ℤ → ℤ) (h₀ : ∀ x, f x = abs (x + 4))
  (h₁ : ∀ x, x ∈ s ↔ f x < 9) : s.card = 17 := by
  have hs : s = Finset.Icc (-12) 4 := by
    ext x
    simp only [Finset.mem_Icc, h₁, h₀, abs_lt]
    omega
  rw [hs]
  decide

import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_668 (l r : ZMod 7) (h₀ : l = (2 + 3)⁻¹) (h₁ : r = 2⁻¹ + 3⁻¹) :
  l - r = 1 := by
  subst h₀; subst h₁; native_decide

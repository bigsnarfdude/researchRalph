import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_326 (n : ℤ) (h₀ : (n - 1) * n * (n + 1) = 720 ) : n + 1 = 10 := by
  have : n = 9 := by nlinarith [sq_nonneg (n-9), sq_nonneg (n+9)]
  linarith

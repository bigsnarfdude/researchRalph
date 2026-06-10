import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_461 (n : ℕ)
  (h₀ : n = Finset.card (Finset.filter (fun x => Nat.gcd x 8 = 1) (Finset.Icc 1 7))) :
  3 ^ n % 8 = 1 := by
  have hn : n = 4 := by rw [h₀]; native_decide
  rw [hn]; norm_num

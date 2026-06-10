import Mathlib
set_option maxHeartbeats 1600000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_64 : IsLeast { x : ℕ | 30 * x ≡ 42 [MOD 47] } 39 := by
  refine ⟨?_, fun x hx => ?_⟩
  · show 30 * 39 % 47 = 42 % 47; omega
  · simp only [Set.mem_setOf_eq, Nat.ModEq] at hx; omega

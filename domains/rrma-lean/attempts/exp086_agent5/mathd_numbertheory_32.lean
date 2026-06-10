import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_32 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ 36) : (∑ k ∈ S, k) = 91 := by
  have : S = Nat.divisors 36 := by ext n; rw [h₀, Nat.mem_divisors]; simp [show 36 ≠ 0 from by norm_num]
  rw [this]; native_decide

import Mathlib
set_option maxHeartbeats 400000
open BigOperators Real Nat Topology Rat
theorem mathd_numbertheory_35 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ Nat.sqrt 196) :
    (∑ k ∈ S, k) = 24 := by
  have h14 : Nat.sqrt 196 = 14 := by native_decide
  rw [h14] at h₀
  have : S = Nat.divisors 14 := by ext n; rw [h₀, Nat.mem_divisors]; simp [show 14 ≠ 0 from by norm_num]
  rw [this]; native_decide

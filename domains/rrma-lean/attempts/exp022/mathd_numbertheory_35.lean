import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_35 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ Nat.sqrt 196) :
    (∑ k ∈ S, k) = 24 := by
  have hsqrt : Nat.sqrt 196 = 14 := by native_decide
  have hS : S = Nat.divisors 14 := by
    ext n
    simp [h₀, Nat.mem_divisors, hsqrt]
  rw [hS]
  native_decide

import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_32 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ 36) : (∑ k ∈ S, k) = 91 := by
  have hS : S = Nat.divisors 36 := by
    ext n
    simp [h₀, Nat.mem_divisors]
  rw [hS]
  native_decide

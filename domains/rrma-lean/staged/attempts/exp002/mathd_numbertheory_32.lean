import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_32 (S : Finset ℕ) (h₀ : ∀ n : ℕ, n ∈ S ↔ n ∣ 36) : (∑ k ∈ S, k) = 91 := by
  constructor <;> (first
    | intro h; first | omega | linarith | simp_all [*] | exact h
    | intro h; omega
    | intro h; linarith
    | intro; simp_all [*]
    | intro; norm_num)
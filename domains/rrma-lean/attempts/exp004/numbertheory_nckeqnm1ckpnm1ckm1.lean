import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_nckeqnm1ckpnm1ckm1 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₁ : k ≤ n) :
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) := by
  first
    | omega
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide
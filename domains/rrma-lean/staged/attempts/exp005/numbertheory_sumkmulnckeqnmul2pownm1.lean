import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sumkmulnckeqnmul2pownm1 (n : ℕ) (h₀ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, k * Nat.choose n k) = n * 2 ^ (n - 1) := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
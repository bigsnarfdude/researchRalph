import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_sumkmulnckeqnmul2pownm1 (n : ℕ) (h₀ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, k * Nat.choose n k) = n * 2 ^ (n - 1) := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg h₀, sq_nonneg (n - h₀), sq_nonneg (n + h₀), mul_self_nonneg (n - h₀)]
    | simp_all [*]
    | decide
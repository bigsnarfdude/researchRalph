import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_nckeqnm1ckpnm1ckm1 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₁ : k ≤ n) :
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg k, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (n - k), sq_nonneg (n + k), mul_self_nonneg (n - k)]
    | simp_all
    | decide)
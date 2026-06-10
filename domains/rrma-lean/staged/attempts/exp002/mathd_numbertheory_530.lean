import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_530 (n k : ℕ) (h₀ : 0 < n ∧ 0 < k) (h₀ : (n : ℝ) / k < 6)
  (h₁ : (5 : ℝ) < n / k) : 22 ≤ Nat.lcm n k / Nat.gcd n k := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg k, sq_nonneg h₀, sq_nonneg h₀, sq_nonneg (n - k), sq_nonneg (n + k), mul_self_nonneg (n - k)]
    | simp_all
    | decide)
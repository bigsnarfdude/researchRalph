import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1988_p4 (n : ℕ) (a : ℕ → ℝ) (h₀ : ∀ n, abs (a n) < 1)
  (h₁ : (∑ k ∈ Finset.range n, abs (a k)) = 19 + abs (∑ k ∈ Finset.range n, a k)) : 20 ≤ n := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg a, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (n - a), sq_nonneg (n + a), mul_self_nonneg (n - a)]
    | simp_all [*]
    | decide
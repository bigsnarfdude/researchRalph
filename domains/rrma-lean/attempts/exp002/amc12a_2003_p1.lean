import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p1 (u v : ℕ → ℕ) (h₀ : ∀ n, u n = 2 * n + 2) (h₁ : ∀ n, v n = 2 * n + 1) :
    ((∑ k ∈ Finset.range 2003, u k) - ∑ k ∈ Finset.range 2003, v k) = 2003 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg u, sq_nonneg v, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (u - v), sq_nonneg (u + v), mul_self_nonneg (u - v)]
    | simp_all [*]
    | decide
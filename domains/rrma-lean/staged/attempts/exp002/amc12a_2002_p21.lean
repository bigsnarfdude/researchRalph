import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p21 (u : ℕ → ℕ) (h₀ : u 0 = 4) (h₁ : u 1 = 7)
    (h₂ : ∀ n ≥ 2, u (n + 2) = (u n + u (n + 1)) % 10) :
    ∀ n, (∑ k ∈ Finset.range n, u k) > 10000 → 1999 ≤ n := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg u, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg h₂, sq_nonneg (u - h₀), sq_nonneg (u + h₀), mul_self_nonneg (u - h₀)]
    | simp_all [*]
    | decide
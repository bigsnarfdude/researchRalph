import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem induction_seq_mul2pnp1 (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + (n + 1)) : u n = 2 ^ (n + 1) - (n + 2) := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg n, sq_nonneg u, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (n - u), sq_nonneg (n + u), mul_self_nonneg (n - u)]
    | simp_all [*]
    | decide
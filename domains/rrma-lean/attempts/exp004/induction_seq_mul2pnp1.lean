import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem induction_seq_mul2pnp1 (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + (n + 1)) : u n = 2 ^ (n + 1) - (n + 2) := by
  first
    | omega
    | simp only [h₁]; ring
    | simp only [h₁]; norm_num
    | ring
    | norm_num
    | linarith
    | simp_all
    | decide
import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_13 (u v : ℕ) (S : Set ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ 14 * n % 100 = 46) (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v) : (u + v : ℚ) / 2 = 64 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg u, sq_nonneg v, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (u - v), sq_nonneg (u + v), mul_self_nonneg (u - v)]
    | simp_all
    | decide)
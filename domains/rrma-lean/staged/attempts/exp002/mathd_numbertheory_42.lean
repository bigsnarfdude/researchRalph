import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_42 (S : Set ℕ) (u v : ℕ) (h₀ : ∀ a : ℕ, a ∈ S ↔ 0 < a ∧ 27 * a % 40 = 17)
    (h₁ : IsLeast S u) (h₂ : IsLeast (S \ {u}) v) : u + v = 62 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg u, sq_nonneg v, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (u - v), sq_nonneg (u + v), mul_self_nonneg (u - v)]
    | simp_all
    | decide)
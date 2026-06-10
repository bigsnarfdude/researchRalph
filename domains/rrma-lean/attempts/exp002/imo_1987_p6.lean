import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg p, sq_nonneg f, sq_nonneg h₀, sq_nonneg h₀, sq_nonneg (p - f), sq_nonneg (p + f), mul_self_nonneg (p - f)]
    | simp_all [*]
    | decide
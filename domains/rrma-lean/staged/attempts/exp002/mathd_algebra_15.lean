import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_15 (s : ℕ → ℕ → ℕ)
    (h₀ : ∀ a b, 0 < a ∧ 0 < b → s a b = a ^ (b : ℕ) + b ^ (a : ℕ)) : s 2 6 = 100 := by
  constructor <;> (first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg s, sq_nonneg h₀, sq_nonneg a, sq_nonneg (s - h₀), sq_nonneg (s + h₀), mul_self_nonneg (s - h₀)]
    | simp_all
    | decide)
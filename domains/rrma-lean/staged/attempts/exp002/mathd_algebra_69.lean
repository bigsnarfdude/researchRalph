import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_69 (rows seats : ℕ) (h₀ : rows * seats = 450)
  (h₁ : (rows + 5) * (seats - 3) = 450) : rows = 25 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (h₀ - h₁), sq_nonneg (h₀ + h₁), mul_self_nonneg (h₀ - h₁)]
    | simp_all [*]
    | decide
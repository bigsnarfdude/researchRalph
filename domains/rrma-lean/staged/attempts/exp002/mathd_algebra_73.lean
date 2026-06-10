import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_73 (p q r x : ℂ) (h₀ : (x - p) * (x - q) = (r - p) * (r - q)) (h₁ : x ≠ r) :
  x = p + q - r := by
  first
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg p, sq_nonneg q, sq_nonneg r, sq_nonneg x, sq_nonneg (p - q), sq_nonneg (p + q), mul_self_nonneg (p - q)]
    | simp_all [*]
    | decide
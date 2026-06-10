import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1979_p1 (p q : ℕ) (h₀ : 0 < q)
  (h₁ : (∑ k ∈ Finset.Icc (1 : ℕ) 1319, (-1) ^ (k + 1) * ((1 : ℝ) / k)) = p / q) : 1979 ∣ p := by
  first
    | omega
    | field_simp; ring
    | field_simp; linarith
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg p, sq_nonneg q, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (p - q), sq_nonneg (p + q), mul_self_nonneg (p - q)]
    | simp_all [*]
    | decide
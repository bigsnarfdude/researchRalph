import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_109 (v : ℕ → ℕ) (h₀ : ∀ n, v n = 2 * n - 1) :
  (∑ k ∈ Finset.Icc 1 100, v k) % 7 = 4 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg v, sq_nonneg h₀, sq_nonneg (v - h₀), sq_nonneg (v + h₀), mul_self_nonneg (v - h₀)]
    | simp_all [*]
    | decide
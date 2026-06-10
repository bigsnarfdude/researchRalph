import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem numbertheory_aneqprodakp4_anmsqrtanp1eq2 (a : ℕ → ℝ) (h₀ : a 0 = 1)
  (h₁ : ∀ n, a (n + 1) = (∏ k ∈ Finset.range (n + 1), a k) + 4) :
  ∀ n ≥ 1, a n - Real.sqrt (a (n + 1)) = 2 := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg a, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (a - h₀), sq_nonneg (a + h₀), mul_self_nonneg (a - h₀)]
    | simp_all [*]
    | decide
import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_31 (x : NNReal) (u : ℕ → NNReal) (h₀ : ∀ n, u (n + 1) = NNReal.sqrt (x + u n))
  (h₁ : Filter.Tendsto u Filter.atTop (𝓝 9)) : 9 = NNReal.sqrt (x + 9) := by
  first
    | omega
    | omega
    | linarith
    | ring
    | norm_num
    | nlinarith [sq_nonneg x, sq_nonneg u, sq_nonneg h₀, sq_nonneg h₁, sq_nonneg (x - u), sq_nonneg (x + u), mul_self_nonneg (x - u)]
    | simp_all [*]
    | decide
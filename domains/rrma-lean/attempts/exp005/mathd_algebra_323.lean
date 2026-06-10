import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_323 (σ : Equiv ℝ ℝ) (h : ∀ x, σ.1 x = x ^ 3 - 8) : σ.2 (σ.1 (σ.2 19)) = 3 := by
  first
    | simp only [h] at *; nlinarith
    | simp only [h] at *; linarith
    | simp only [h] at *; omega
    | simp only [h] at *; norm_num
    | simp only [h]; ring
    | simp only [h]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
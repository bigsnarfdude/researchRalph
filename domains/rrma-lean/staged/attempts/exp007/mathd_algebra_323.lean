import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_algebra_323 (σ : Equiv ℝ ℝ) (h : ∀ x, σ.1 x = x ^ 3 - 8) : σ.2 (σ.1 (σ.2 19)) = 3 := by
  first
  | solve | simp only [h] at *; ring
  | solve | simp only [h] at *; norm_num
  | solve | simp only [h] at *; omega
  | solve | simp only [h] at *; linarith
  | solve | simp only [h] at *; nlinarith
  | solve | linear_combination h
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
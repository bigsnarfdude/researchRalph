import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p21 (u : ℕ → ℕ) (h₀ : u 0 = 4) (h₁ : u 1 = 7)
    (h₂ : ∀ n ≥ 2, u (n + 2) = (u n + u (n + 1)) % 10) :
    ∀ n, (∑ k ∈ Finset.range n, u k) > 10000 → 1999 ≤ n := by
  first
  | solve | simp only [h₂] at *; ring
  | solve | simp only [h₂] at *; norm_num
  | solve | simp only [h₂] at *; omega
  | solve | simp only [h₂] at *; linarith
  | solve | simp only [h₂] at *; nlinarith
  | solve | simp only [h₂]; norm_num
  | solve | simp only [h₂]; omega
  | solve | linarith [h₀, h₁, h₂]
  | solve | nlinarith [h₀, h₁, h₂]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | native_decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
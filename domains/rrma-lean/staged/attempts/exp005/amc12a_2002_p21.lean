import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2002_p21 (u : ℕ → ℕ) (h₀ : u 0 = 4) (h₁ : u 1 = 7)
    (h₂ : ∀ n ≥ 2, u (n + 2) = (u n + u (n + 1)) % 10) :
    ∀ n, (∑ k ∈ Finset.range n, u k) > 10000 → 1999 ≤ n := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | simp only [h₂] at *; nlinarith
    | simp only [h₂] at *; linarith
    | simp only [h₂] at *; omega
    | simp only [h₂] at *; norm_num
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | ring
    | norm_num
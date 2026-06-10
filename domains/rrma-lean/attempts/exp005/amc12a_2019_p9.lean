import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2019_p9 (a : ℕ → ℚ) (h₀ : a 1 = 1) (h₁ : a 2 = 3 / 7)
  (h₂ : ∀ n, a (n + 2) = a n * a (n + 1) / (2 * a n - a (n + 1))) :
  ↑(a 2019).den + (a 2019).num = 8078 := by
  first
    | simp only [h₂] at *; nlinarith
    | simp only [h₂] at *; linarith
    | simp only [h₂] at *; omega
    | simp only [h₂] at *; norm_num
    | simp only [h₂]; ring
    | simp only [h₂]; norm_num
    | ring
    | norm_num
    | omega
    | linarith
    | simp_all
    | decide
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_13 (u v : ℕ) (S : Set ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ 14 * n % 100 = 46) (h₁ : IsLeast S u)
  (h₂ : IsLeast (S \ {u}) v) : (u + v : ℚ) / 2 = 64 := by
  first
    | omega
    | simp only [h₀] at *; nlinarith
    | simp only [h₀] at *; linarith
    | simp only [h₀] at *; omega
    | simp only [h₀] at *; norm_num
    | simp only [h₀] at *; field_simp; ring
    | simp only [h₀] at *; field_simp; linarith
    | simp only [h₀]; ring
    | simp only [h₀]; norm_num
    | ring
    | norm_num
    | linarith
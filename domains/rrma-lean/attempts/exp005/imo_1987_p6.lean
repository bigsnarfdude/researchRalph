import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  first
    | omega
    | norm_num
    | native_decide
    | decide
    | simp only [h₀, h₀] at *; nlinarith
    | simp only [h₀, h₀] at *; linarith
    | simp only [h₀, h₀] at *; omega
    | simp only [h₀, h₀] at *; norm_num
    | simp only [h₀, h₀]; ring
    | simp only [h₀, h₀]; norm_num
    | ring
    | linarith
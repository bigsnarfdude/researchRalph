import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  first
  | solve | norm_num
  | solve | simp only [h₀, h₀] at *; ring
  | solve | simp only [h₀, h₀] at *; norm_num
  | solve | simp only [h₀, h₀] at *; omega
  | solve | simp only [h₀, h₀] at *; linarith
  | solve | simp only [h₀, h₀] at *; nlinarith
  | solve | simp only [h₀, h₀] at *; field_simp; ring
  | solve | simp only [h₀, h₀] at *; field_simp; linarith
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
  | solve | nlinarith [sq_nonneg _, h₀, h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
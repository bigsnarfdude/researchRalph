import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  first
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
  | solve | nlinarith [sq_nonneg p, h₀, h₀]
  | solve | nlinarith [sq_nonneg (p - 1), h₀, h₀]
  | solve | omega
  | solve | field_simp; nlinarith [h₀, h₀]
  | solve | field_simp; linarith [h₀, h₀]
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
  | solve | simp only [h₀, h₀]; ring
  | solve | simp only [h₀, h₀]; norm_num
  | solve | simp only [h₀, h₀]; omega
  | solve | simp only [h₀, h₀]; linarith
  | solve | simp only [h₀, h₀]; nlinarith
  | solve | push_cast; ring
  | solve | push_cast; norm_num
  | solve | push_cast; omega
  | solve | field_simp; ring
  | solve | field_simp; norm_num
  | solve | ring_nf; norm_num
  | solve | ring_nf; omega
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
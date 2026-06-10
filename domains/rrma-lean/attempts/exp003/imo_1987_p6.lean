import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1987_p6 (p : ℕ) (f : ℕ → ℕ) (h₀ : ∀ x, f x = x ^ 2 + x + p)
  (h₀ : ∀ k : ℕ, k ≤ Nat.floor (Real.sqrt (p / 3)) → Nat.Prime (f k)) :
   ∀ i ≤ p - 2, Nat.Prime (f i) := by
  first
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | omega
  | solve | simp only [h₀, h₀]
  | solve | simp only [h₀, h₀]; ring
  | solve | simp only [h₀, h₀]; norm_num
  | solve | simp only [h₀, h₀]; linarith
  | solve | simp only [h₀, h₀]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₀]
  | solve | nlinarith [h₀, h₀]
  | solve | norm_num
  | solve | ring
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num
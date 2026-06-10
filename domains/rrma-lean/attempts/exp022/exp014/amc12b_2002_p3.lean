import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12b_2002_p3 (S : Finset ℕ)
  -- note: we use (n^2 + 2 - 3 * n) over (n^2 - 3 * n + 2) because nat subtraction truncates the latter at 1 and 2
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 0 < n ∧ Nat.Prime (n ^ 2 + 2 - 3 * n)) :
  S.card = 1 := by
  first
  | solve | norm_num
  | solve | simp only [h₀] at *; ring
  | solve | simp only [h₀] at *; norm_num
  | solve | simp only [h₀] at *; omega
  | solve | simp only [h₀] at *; linarith
  | solve | simp only [h₀] at *; nlinarith
  | solve | simp only [h₀] at *; constructor <;> (first | norm_num | omega | linarith | nlinarith)
  | solve | simp only [h₀]; norm_num
  | solve | simp only [h₀]; omega
  | solve | constructor <;> intro <;> omega
  | solve | constructor <;> intro <;> linarith
  | solve | constructor <;> (intro; simp_all)
  | solve | linarith [h₀]
  | solve | nlinarith [h₀]
  | solve | nlinarith [sq_nonneg _, h₀]
  | solve | linarith
  | solve | nlinarith
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | omega
  | solve | ring
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
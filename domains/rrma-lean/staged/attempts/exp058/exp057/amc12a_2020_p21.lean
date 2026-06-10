import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2020_p21 (S : Finset ℕ)
  (h₀ : ∀ n : ℕ, n ∈ S ↔ 5 ∣ n ∧ Nat.lcm 5! n = 5 * Nat.gcd 10! n) : S.card = 48 := by
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
  | solve | omega
  | solve | simp; omega
  | solve | native_decide
  | solve | decide
  | solve | simp_all
  | solve | simp_all; ring
  | solve | simp_all; omega
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp; ring
  | solve | simp; norm_num
  | solve | push_cast; ring
  | solve | push_cast; norm_num
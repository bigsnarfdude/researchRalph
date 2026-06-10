import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1984_p2 (a b : ℤ) (h₀ : 0 < a ∧ 0 < b) (h₁ : ¬7 ∣ a) (h₂ : ¬7 ∣ b) (h₃ : ¬7 ∣ a + b)
  (h₄ : 7 ^ 7 ∣ (a + b) ^ 7 - a ^ 7 - b ^ 7) : 19 ≤ a + b := by
  first
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄]
  | solve | omega
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
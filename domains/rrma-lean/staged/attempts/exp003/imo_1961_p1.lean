import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem imo_1961_p1 (x y z a b : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x ≠ y) (h₂ : y ≠ z)
  (h₃ : z ≠ x) (h₄ : x + y + z = a) (h₅ : x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆ : x * y = z ^ 2) :
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
  first
  | solve | constructor <;> linarith
  | solve | constructor <;> nlinarith
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | nlinarith [sq_nonneg (a - b), sq_nonneg a, sq_nonneg b]
  | solve | nlinarith [sq_nonneg (_ - _)]
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
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
import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1961_p1 (x y z a b : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x ≠ y) (h₂ : y ≠ z)
  (h₃ : z ≠ x) (h₄ : x + y + z = a) (h₅ : x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆ : x * y = z ^ 2) :
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
  first
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | constructor <;> linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | constructor <;> omega
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | decide
  | solve | native_decide
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; nlinarith
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
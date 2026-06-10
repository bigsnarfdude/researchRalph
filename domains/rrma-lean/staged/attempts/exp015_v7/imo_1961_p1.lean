import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1961_p1 (x y z a b : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x ≠ y) (h₂ : y ≠ z)
  (h₃ : z ≠ x) (h₄ : x + y + z = a) (h₅ : x ^ 2 + y ^ 2 + z ^ 2 = b ^ 2) (h₆ : x * y = z ^ 2) :
  0 < a ∧ b ^ 2 < a ^ 2 ∧ a ^ 2 < 3 * b ^ 2 := by
  first
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | omega
  | solve | norm_num
  | solve | ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp
  | solve | simp_all
  | solve | native_decide
  | solve | decide
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; ring
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; norm_num
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; omega
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; omega
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; linarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; nlinarith
  | solve | simp [h₀, h₁, h₂, h₃, h₄, h₅, h₆]; nlinarith
  | solve | linear_combination h₀
  | solve | linear_combination h₁
  | solve | linear_combination h₂
  | solve | linear_combination h₀ + h₁
  | solve | linear_combination h₀ + -h₁
  | solve | linear_combination -h₀ + h₁
  | solve | linear_combination 2 * h₀ + -h₁
  | solve | linear_combination -h₀ + 2 * h₁
  | solve | linear_combination 3 * h₀ + -h₁
  | solve | linear_combination -3 * h₀ + 2 * h₁
  | solve | constructor <;> linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | constructor <;> nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆]
  | solve | constructor <;> omega
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | constructor <;> simp
  | solve | ring_nf; omega
  | solve | ring_nf; norm_num
  | solve | ring_nf; ring
  | solve | ring_nf; linarith
  | solve | ring_nf; nlinarith
  | solve | ring_nf; simp
  | solve | simp_all; omega
  | solve | simp_all; norm_num
  | solve | simp_all; ring
  | solve | simp_all; linarith
  | solve | simp_all; nlinarith
  | solve | simp_all; simp
  | solve | push_cast; omega
  | solve | push_cast; norm_num
  | solve | push_cast; ring
  | solve | push_cast; linarith
  | solve | push_cast; nlinarith
  | solve | push_cast; simp
  | solve | norm_cast; omega
  | solve | norm_cast; norm_num
  | solve | norm_cast; ring
  | solve | norm_cast; linarith
  | solve | norm_cast; nlinarith
  | solve | norm_cast; simp
  | solve | field_simp; omega
  | solve | field_simp; norm_num
  | solve | field_simp; ring
  | solve | field_simp; linarith
  | solve | field_simp; nlinarith
  | solve | field_simp; simp
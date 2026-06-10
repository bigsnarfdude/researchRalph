import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aimeI_2000_p7 (x y z : ℝ) (m : ℚ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x * y * z = 1)
  (h₂ : x + 1 / z = 5) (h₃ : y + 1 / x = 29) (h₄ : z + 1 / y = m) (h₅ : 0 < m) :
  ↑m.den + m.num = 5 := by
  first
  | solve | constructor <;> linarith
  | solve | constructor <;> nlinarith
  | solve | constructor <;> norm_num
  | solve | constructor <;> ring
  | solve | linarith
  | solve | nlinarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅]
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
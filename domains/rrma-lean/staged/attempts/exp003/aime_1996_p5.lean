import Mathlib

set_option maxHeartbeats 800000

open BigOperators Real Nat Topology Rat

theorem aime_1996_p5 (a b c r s t : ℝ) (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x ^ 3 + 3 * x ^ 2 + 4 * x - 11) (h₁ : ∀ x, g x = x ^ 3 + r * x ^ 2 + s * x + t)
  (h₂ : f a = 0) (h₃ : f b = 0) (h₄ : f c = 0) (h₅ : g (a + b) = 0) (h₆ : g (b + c) = 0)
  (h₇ : g (c + a) = 0) (h₈ : List.Pairwise (· ≠ ·) [a, b, c]) : t = 23 := by
  first
  | solve | ring
  | solve | norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; ring
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; norm_num
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; linarith
  | solve | simp only [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]; omega
  | solve | subst_vars; ring
  | solve | subst_vars; norm_num
  | solve | subst_vars; omega
  | solve | subst_vars; simp
  | solve | linarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]
  | solve | nlinarith [h₀, h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈]
  | solve | omega
  | solve | linarith
  | solve | nlinarith
  | solve | decide
  | solve | simp
  | solve | simp; ring
  | solve | simp; omega
  | solve | simp; norm_num
  | solve | simp; linarith
  | solve | norm_num; omega
  | solve | push_cast; ring
  | solve | push_cast; norm_num
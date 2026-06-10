import Mathlib
set_option maxHeartbeats 16000000
open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p25 (a b : ℝ) (f : ℝ → ℝ) (h₀ : 0 < b)
  (h₁ : ∀ x, f x = Real.sqrt (a * x ^ 2 + b * x)) (h₂ : { x | 0 ≤ f x } = f '' { x | 0 ≤ f x }) :
  a = 0 ∨ a = -4 := by
  exfalso
  have hmem : (-1 : ℝ) ∈ { x : ℝ | 0 ≤ f x } := by
    simp only [Set.mem_setOf_eq]
    rw [h₁]
    exact Real.sqrt_nonneg _
  have hnotmem : (-1 : ℝ) ∉ f '' { x : ℝ | 0 ≤ f x } := by
    intro ⟨y, _, hfy⟩
    have : f y = -1 := hfy
    rw [h₁] at this
    have hsqrt := Real.sqrt_nonneg (a * y ^ 2 + b * y)
    linarith
  rw [h₂] at hmem
  exact hnotmem hmem

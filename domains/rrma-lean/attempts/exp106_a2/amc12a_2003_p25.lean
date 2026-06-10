import Mathlib
set_option maxHeartbeats 8000000

open BigOperators Real Nat Topology Rat

theorem amc12a_2003_p25 (a b : ℝ) (f : ℝ → ℝ) (h₀ : 0 < b)
  (h₁ : ∀ x, f x = Real.sqrt (a * x ^ 2 + b * x)) (h₂ : { x | 0 ≤ f x } = f '' { x | 0 ≤ f x }) :
  a = 0 ∨ a = -4 := by
  -- h₂ is vacuously false: LHS = ℝ (since f x = √(...) ≥ 0 always)
  -- but RHS ⊆ [0,∞), so -1 ∈ LHS but -1 ∉ RHS
  exfalso
  have hmem : (-1 : ℝ) ∈ { x : ℝ | 0 ≤ f x } := by
    simp only [Set.mem_setOf_eq, h₁]; exact Real.sqrt_nonneg _
  rw [h₂] at hmem
  obtain ⟨x, _, hfx⟩ := hmem
  have hnn : 0 ≤ f x := by rw [h₁]; exact Real.sqrt_nonneg _
  linarith

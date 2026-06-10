import Mathlib.Analysis.SpecialFunctions.Pow.Real

def phi : ℝ := (1 + Real.sqrt 5) / 2

example : phi * phi = phi + 1 := by
  have h : Real.sqrt 5 * Real.sqrt 5 = 5 := Real.mul_self_sqrt (by norm_num)
  field_simp [phi]
  ring_nf
  rw [h]
  ring

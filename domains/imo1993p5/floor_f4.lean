import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Irrational

open Real Nat

noncomputable section

def phi : ℝ := (1 + sqrt 5) / 2

def f_floor (n : ℕ) : ℕ :=
  Int.toNat ⌊(n : ℝ) * phi + 0.5⌋

theorem floor_4 : f_floor 4 = 6 := by
  unfold f_floor phi
  have h1 : 6.5 ≤ (4 : ℝ) * ((1 + sqrt 5) / 2) + 0.5 := by
    rw [mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)]
    rw [mul_comm, ← mul_assoc, show (2 : ℝ) * 1 = 2 by norm_num, add_mul, mul_one]
    apply (le_sub_iff_add_le).mpr
    rw [show (6.5 : ℝ) - 0.5 - 2 = 4 by norm_num]
    apply (le_div_iff (by norm_num)).mpr
    rw [show (4 : ℝ) * 1 = 4 by norm_num]
    apply (Real.le_sqrt (by norm_num) (by norm_num)).mpr
    norm_num
  have h2 : (4 : ℝ) * ((1 + sqrt 5) / 2) + 0.5 < 7.5 := by
    rw [mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)]
    rw [mul_comm, ← mul_assoc, show (2 : ℝ) * 1 = 2 by norm_num, add_mul, mul_one]
    apply (lt_sub_iff_add_lt).mpr
    rw [show (7.5 : ℝ) - 0.5 - 2 = 5 by norm_num]
    apply (lt_div_iff (by norm_num)).mpr
    rw [show (5 : ℝ) * 1 = 5 by norm_num]
    apply (Real.sqrt_lt (by norm_num) (by norm_num)).mpr
    norm_num
  have : ⌊(4 : ℝ) * phi + 0.5⌋ = 6 := by
    apply Int.floor_eq_iff.mpr
    constructor
    · linarith
    · linarith
  rw [this]
  norm_num

import Mathlib.NumberTheory.Rayleigh
import Mathlib.RingTheory.Real.Irrational
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Real Nat

noncomputable section

def phi : ℝ := (1 + Real.sqrt 5) / 2

lemma phi_pos : 0 < phi := by
  unfold phi
  have : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num)
  positivity

lemma phi_gt_one : 1 < phi := by
  unfold phi
  rw [div_lt_iff (by norm_num)]
  have : 1 < Real.sqrt 5 := by
    rw [← Real.sqrt_one, Real.sqrt_lt_sqrt_iff (by norm_num)]
    norm_num
  linarith

lemma phi_sq : phi^2 = phi + 1 := by
  unfold phi
  have h5 : (Real.sqrt 5)^2 = 5 := Real.sq_sqrt (by norm_num)
  ring_nf
  rw [h5]
  ring

lemma phi_inv_sq : 1 / phi^2 = 2 - phi := by
  have h_phi : phi ≠ 0 := phi_pos.ne'
  field_simp [phi_sq]
  ring

lemma phi_inv : 1 / phi = phi - 1 := by
  have h_phi : phi ≠ 0 := phi_pos.ne'
  field_simp [phi_sq]
  ring

def f_c (c : ℝ) (n : ℕ) : ℕ :=
  Int.toNat ⌊(n : ℝ) * phi + c⌋

theorem f_c_functional (c : ℝ) (hc1 : 1 / phi^2 ≤ c) (hc2 : c ≤ 1 / phi)
    (h_ir : ∀ n > 0, Int.fract ((n : ℝ) * phi + c) ≠ 0) (n : ℕ) (hn : 0 < n) :
    f_c c (f_c c n) = f_c c n + n := by
  set m := f_c c n
  have h_m_nonneg : 0 ≤ ⌊(n : ℝ) * phi + c⌋ := by
    apply Int.floor_nonneg.mpr
    calc (0 : ℝ) ≤ 1 * phi + 1 / phi^2 := by positivity
      _ ≤ (n : ℝ) * phi + c := by
        apply add_le_add
        · apply mul_le_mul_of_nonneg_right; exact_mod_cast hn; exact phi_pos.le
        · exact hc1
  have h_m : (m : ℤ) = ⌊(n : ℝ) * phi + c⌋ := by
    unfold f_c
    rw [Int.toNat_of_nonneg h_m_nonneg]
  have h_m_val : (m : ℝ) = ⌊(n : ℝ) * phi + c⌋ := by rw [← h_m]; norm_cast
  
  have h_fm_nonneg : 0 ≤ ⌊(m : ℝ) * phi + c⌋ := by
    apply Int.floor_nonneg.mpr
    calc (0 : ℝ) ≤ 1 * phi + 1 / phi^2 := by positivity
      _ ≤ (m : ℝ) * phi + c := by
        apply add_le_add
        · apply mul_le_mul_of_nonneg_right
          · rw [h_m_val]
            apply Int.le_floor.mpr
            calc (1 : ℝ) ≤ 1 * phi + 1 / phi^2 := by positivity
              _ ≤ (n : ℝ) * phi + c := by
                apply add_le_add
                · apply mul_le_mul_of_nonneg_right; exact_mod_cast hn; exact phi_pos.le
                · exact hc1
          · exact phi_pos.le
        · exact hc1

  unfold f_c
  rw [Int.toNat_of_nonneg h_fm_nonneg]
  rw [show (f_c c n + n : ℤ) = m + n by omega]
  apply Int.floor_eq_iff.mpr
  constructor
  · -- m + n ≤ m * phi + c
    rw [h_m_val]
    set x := (n : ℝ) * phi + c
    have h_x_eq : (n : ℝ) * phi = x - c := by ring
    rw [Int.floor_eq_sub_fract]
    set fract := Int.fract x
    have h_phi_inv : 1 / phi = phi - 1 := phi_inv
    have : (n : ℝ) = (x - c) * (phi - 1) := by
      rw [← h_x_eq, mul_comm, ← mul_assoc, ← phi_inv, mul_div_cancel₀ _ phi_pos.ne']
    calc (⌊x⌋ + n : ℝ) = ⌊x⌋ + (⌊x⌋ + fract - c) * (phi - 1) := by rw [this, Int.floor_add_fract]
      _ = ⌊x⌋ + ⌊x⌋ * (phi - 1) + (fract - c) * (phi - 1) := by ring
      _ = ⌊x⌋ * phi + (fract - c) * (phi - 1) := by ring
    rw [show phi - 1 = 1 / phi from h_phi_inv.symm]
    rw [le_div_iff phi_pos]
    rw [show c * phi = c * (phi^2 - 1) from by rw [phi_sq]; ring]
    rw [sub_mul, one_mul, le_sub_iff_add_le]
    calc fract ≤ (1 : ℝ) := (Int.fract_lt_one x).le
      _ ≤ 1 / phi^2 * phi^2 := by field_simp [phi_pos.ne']
      _ ≤ c * phi^2 := (mul_le_mul_of_nonneg_right hc1 (by positivity))
  · -- m * phi + c < m + n + 1
    rw [h_m_val]
    set x := (n : ℝ) * phi + c
    have h_x_eq : (n : ℝ) * phi = x - c := by ring
    rw [Int.floor_eq_sub_fract]
    set fract := Int.fract x
    have h_phi_inv : 1 / phi = phi - 1 := phi_inv
    have : (n : ℝ) = (x - c) * (phi - 1) := by
      rw [← h_x_eq, mul_comm, ← mul_assoc, ← phi_inv, mul_div_cancel₀ _ phi_pos.ne']
    rw [show (m + n + 1 : ℝ) = ⌊x⌋ + (⌊x⌋ + fract - c) * (phi - 1) + 1 from by rw [this, Int.floor_add_fract]]
    rw [show ⌊x⌋ + ⌊x⌋ * (phi - 1) = ⌊x⌋ * phi by ring]
    rw [add_assoc]
    apply add_lt_add_left
    rw [sub_mul, show (fract - c) * (phi - 1) = fract / phi - c / phi by rw [h_phi_inv.symm]; ring]
    rw [lt_sub_iff_add_lt, show c + c / phi = c * (1 + 1 / phi) by ring]
    rw [show 1 + 1 / phi = phi by rw [phi_inv]; ring]
    calc c * phi ≤ (1 / phi) * phi := mul_le_mul_of_nonneg_right hc2 phi_pos.le
      _ = 1 := by field_simp [phi_pos.ne']
      _ < fract / phi + 1 := by
        apply lt_add_of_pos_left
        apply div_pos
        · apply (Int.fract_pos).mpr (h_ir n hn)
        · exact phi_pos

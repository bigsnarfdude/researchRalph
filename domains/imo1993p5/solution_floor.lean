import Mathlib

set_option maxHeartbeats 0

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

private lemma phi_pos : 0 < phi := by
  unfold phi
  apply _root_.div_pos
  · apply add_pos
    · exact zero_lt_one
    · apply Real.sqrt_pos.2
      norm_num
  · norm_num

private lemma one_lt_phi : 1 < phi := by
  unfold phi
  apply (lt_div_iff₀ (by norm_num : (0 : ℝ) < 2)).mpr
  apply (lt_sub_iff_add_lt).mpr
  rw [show (1 : ℝ) * 2 - 1 = 1 by norm_num]
  apply (Real.le_sqrt (by norm_num) (by norm_num)).mpr
  norm_num

private lemma phi_sq : phi^2 = phi + 1 := by
  unfold phi
  have h5 : (Real.sqrt 5)^2 = 5 := Real.sq_sqrt (by norm_num)
  ring_nf
  rw [h5]
  ring

private lemma phi_inv : 1 / phi = phi - 1 := by
  have hphi : phi ≠ 0 := phi_pos.ne'
  field_simp [hphi]
  rw [← sq, phi_sq]
  ring

noncomputable def floor_f (n : ℕ) : ℕ :=
  Int.toNat ⌊(n : ℝ) * phi + 1/2⌋

private lemma floor_f_val (n : ℕ) : 
    (floor_f n : ℝ) ≤ (n : ℝ) * phi + 1 / 2 ∧ (n : ℝ) * phi + 1 / 2 < (floor_f n : ℝ) + 1 := by
  unfold floor_f
  have hpos : 0 ≤ ⌊(n : ℝ) * phi + 1 / 2⌋ := by
    apply Int.floor_nonneg.mpr
    apply add_nonneg
    · apply mul_nonneg
      · exact Nat.cast_nonneg n
      · exact phi_pos.le
    · norm_num
  rw [Int.toNat_of_nonneg hpos]
  constructor
  · exact Int.floor_le ((n : ℝ) * phi + 1 / 2)
  · exact Int.lt_floor_add_one ((n : ℝ) * phi + 1 / 2)

theorem floor_f_one : floor_f 1 = 2 := by
  unfold floor_f phi
  have h1 : 2 ≤ 1 * ((1 + Real.sqrt 5) / 2) + 1 / 2 := by
    rw [mul_one]
    apply (le_div_iff₀ (by norm_num : (0 : ℝ) < 2)).mpr
    rw [add_mul, one_mul, div_mul_cancel (by norm_num : (2 : ℝ) ≠ 0)]
    apply (le_sub_iff_add_le).mpr
    rw [show (2 : ℝ) * 2 - 1 = 3 by norm_num]
    apply (Real.le_sqrt (by norm_num) (by norm_num)).mpr
    norm_num
  have h2 : 1 * ((1 + Real.sqrt 5) / 2) + 1 / 2 < 3 := by
    rw [mul_one]
    apply (div_lt_iff₀ (by norm_num : (0 : ℝ) < 2)).mpr
    rw [add_mul, one_mul, div_mul_cancel (by norm_num : (2 : ℝ) ≠ 0)]
    apply (lt_sub_iff_add_lt).mpr
    rw [show (3 : ℝ) * 2 - 1 = 5 by norm_num]
    apply (Real.sqrt_lt (by norm_num) (by norm_num)).mpr
    norm_num
  rw [Int.toNat_eq_iff (by norm_num)]
  apply Int.floor_eq_iff.mpr
  constructor
  · exact h1
  · exact h2

theorem floor_f_functional (n : ℕ) : floor_f (floor_f n) = floor_f n + n := by
  have hf := floor_f_val n
  have h_low : (floor_f n : ℝ) + (n : ℝ) ≤ (floor_f n : ℝ) * phi + 1 / 2 := by
    rw [show (floor_f n : ℝ) * phi = (floor_f n : ℝ) * (phi - 1) + (floor_f n : ℝ) by ring]
    rw [← phi_inv]
    apply (le_add_iff_le_sub_left).mpr
    rw [add_sub_assoc, show 1 / 2 + (floor_f n : ℝ) - (floor_f n : ℝ) = 1 / 2 by ring]
    apply (le_div_iff₀ phi_pos).mpr
    apply (le_sub_iff_add_le).mp
    rw [sub_add_cancel]
    exact hf.1
  
  have h_high : (floor_f n : ℝ) * phi + 1 / 2 < (floor_f n : ℝ) + (n : ℝ) + 1 := by
    rw [show (floor_f n : ℝ) * phi = (floor_f n : ℝ) * (phi - 1) + (floor_f n : ℝ) by ring]
    rw [← phi_inv]
    apply (add_lt_add_left)
    apply (lt_add_iff_pos_left).mpr
    apply (sub_pos_iff_lt).mpr
    apply (div_lt_iff₀ phi_pos).mpr
    apply (lt_sub_iff_add_lt).mpr
    exact hf.2
  
  unfold floor_f
  have h_nonneg : 0 ≤ ⌊(floor_f n : ℝ) * phi + 1 / 2⌋ := by
    apply Int.floor_nonneg.mpr
    apply add_nonneg
    · apply mul_nonneg
      · apply Nat.cast_nonneg
      · exact phi_pos.le
    · norm_num
  rw [Int.toNat_of_nonneg h_nonneg]
  apply Int.floor_eq_iff.mpr
  constructor
  · exact h_low
  · exact h_high

theorem floor_f_increasing (n : ℕ) : floor_f n < floor_f (n + 1) := by
  unfold floor_f
  apply Int.toNat_lt_toNat
  · apply Int.lt_floor_add_one.mpr
    calc (0 : ℝ) < 1 + 1/2 := by norm_num
      _ ≤ (n + 1 : ℝ) * phi + 1/2 := by
        apply add_le_add_right
        apply (le_mul_of_one_le_right (Nat.cast_nonneg _)).mpr
        apply one_lt_phi.le
  · apply Int.floor_lt_floor_add_one_of_le
    have : ((n + 1 : ℕ) : ℝ) * phi + 1 / 2 = (n : ℝ) * phi + phi + 1 / 2 := by ring
    rw [this]
    apply add_le_add_right
    apply add_le_add_left
    exact one_lt_phi.le

theorem imo_1993_p5_floor :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ (∀ n, f n < f (n + 1)) :=
  ⟨floor_f, floor_f_one, floor_f_functional, floor_f_increasing⟩

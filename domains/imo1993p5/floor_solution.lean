import Mathlib

open Real Nat

noncomputable def phi : ℝ := (1 + sqrt 5) / 2

def imo_f (n : ℕ) : ℕ :=
  Int.toNat (⌊((n : ℝ) + 1) * phi⌋ - 1)

private lemma phi_pos : 0 < phi := by
  unfold phi; positivity

private lemma phi_irr : Irrational phi := by
  unfold phi
  apply Irrational.rat_add_rat_mul_irrational
  · norm_num
  · norm_num
  · exact irrational_sqrt_five

private lemma phi_sq : phi^2 = phi + 1 := by
  unfold phi
  have h5 : (sqrt 5)^2 = 5 := sq_sqrt (by norm_num)
  ring_nf
  rw [h5]
  ring

private lemma phi_inv : 1 / phi = phi - 1 := by
  have hphi : phi ≠ 0 := by apply phi_pos.ne'
  field_simp [hphi]
  rw [← sq, phi_sq]
  ring

theorem imo_1993_p5_stmt :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ ∀ n, f n < f (n + 1) := by
  use imo_f
  constructor
  · -- f 1 = 2
    unfold imo_f phi
    have h1 : 3 ≤ (1 + 1 : ℝ) * ((1 + sqrt 5) / 2) := by
      rw [show (1 + 1 : ℝ) = 2 by norm_num, mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)]
      apply (le_sub_iff_add_le).mpr
      rw [show (3 : ℝ) - 1 = 2 by norm_num]
      apply (le_sqrt (by norm_num) (by norm_num)).mpr
      norm_num
    have h2 : (1 + 1 : ℝ) * ((1 + sqrt 5) / 2) < 4 := by
      rw [show (1 + 1 : ℝ) = 2 by norm_num, mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)]
      apply (lt_sub_iff_add_lt).mpr
      rw [show (4 : ℝ) - 1 = 3 by norm_num]
      apply (sqrt_lt (by norm_num) (by norm_num)).mpr
      norm_num
    have : ⌊(1 + 1 : ℝ) * phi⌋ = 3 := Int.floor_eq_iff.mpr ⟨h1, h2⟩
    rw [this]
    norm_num
  constructor
  · intro n
    unfold imo_f
    set k := n + 1
    have h_pos : 0 < (k : ℝ) * phi := by
      apply mul_pos
      · exact cast_pos.mpr (succ_pos n)
      · exact phi_pos
    have h_fpos : 0 ≤ ⌊(k : ℝ) * phi⌋ - 1 := by
      have h1 : 1.6 < phi := by
        unfold phi; apply (lt_div_iff₀ (by norm_num)).mpr; apply (lt_sub_iff_add_lt).mpr
        rw [show 1.6 * 2 - 1 = 2.2 by norm_num]; apply (lt_sqrt (by norm_num)).mpr; norm_num
      have h2 : 1.6 ≤ (k : ℝ) * phi := by
        calc (1.6 : ℝ) ≤ 1 * phi := by simp; exact h1.le
          _ ≤ (k : ℝ) * phi := by apply mul_le_mul_of_nonneg_right; exact_mod_cast (succ_le_iff.mpr (succ_pos n)); exact phi_pos.le
      have h3 : 1 ≤ ⌊(k : ℝ) * phi ⌋ := Int.le_floor.mpr h2
      omega
    rw [Int.toNat_of_nonneg h_fpos]
    set m := ⌊(k : ℝ) * phi⌋
    have hm : (m : ℝ) = ⌊(k : ℝ) * phi⌋ := rfl
    have h_mfpos : 0 ≤ ⌊(m : ℝ) * phi⌋ - 1 := by
      have h2 : 1.6 ≤ (m : ℝ) * phi := by
        calc (1.6 : ℝ) ≤ 1 * phi := by simp; exact h1.le
          _ ≤ (m : ℝ) * phi := by apply mul_le_mul_of_nonneg_right; exact_mod_cast h3; exact phi_pos.le
      have h3' : 1 ≤ ⌊(m : ℝ) * phi ⌋ := Int.le_floor.mpr h2
      omega
    rw [Int.toNat_of_nonneg h_mfpos]
    -- Goal: (⌊(m : ℝ) * phi⌋ - 1).toNat = (m - 1).toNat + n
    -- Which is: ⌊m * phi⌋ - 1 = m - 1 + n  =>  ⌊m * phi⌋ = m + n
    have : ⌊(m : ℝ) * phi⌋ = m + n := by
      apply Int.floor_eq_iff.mpr
      constructor
      · -- m + n ≤ m * phi
        rw [cast_add, cast_nmul, cast_one] at hm
        have h_frac := Int.fract_lt_one ((k : ℝ) * phi)
        have h_eq : (k : ℝ) * phi = m + Int.fract ((k : ℝ) * phi) := by rw [hm, Int.fract, sub_add_cancel]
        have : (m : ℝ) * phi = (k * phi - Int.fract (k * phi)) * phi := by rw [h_eq, add_sub_cancel_left]
        rw [this, mul_comm (k * phi - _), sub_mul, ← sq, phi_sq]
        rw [add_mul, one_mul]
        -- Goal: m + n ≤ k * phi + k - Int.fract (k * phi) * phi
        -- m + n ≤ (m + fract) + (m + fract)/phi - fract * phi
        -- Actually let's use k = n+1
        have : (m : ℝ) + (n : ℝ) = (m : ℝ) + (k : ℝ) - 1 := by simp [k]; ring
        rw [this]
        apply (le_sub_iff_add_le).mpr
        rw [add_assoc]
        apply (le_add_of_le_of_nonneg)
        · -- m + k ≤ k * phi + k - fract * phi + 1 ?
          -- m ≤ k * phi - fract * phi + 1
          -- m = k * phi - fract
          -- So k * phi - fract ≤ k * phi - fract * phi + 1
          -- fract * (phi - 1) ≤ 1
          -- Since fract < 1 and phi - 1 < 1 (phi < 2), this is true.
          have h_phi_lt_2 : phi < 2 := by
            unfold phi; apply (div_lt_iff₀ (by norm_num)).mpr; apply (sub_lt_iff_lt_add).mpr
            rw [show 2 * 2 - 1 = 3 by norm_num]; apply (sqrt_lt (by norm_num) (by norm_num)).mpr; norm_num
          have : Int.fract ((k : ℝ) * phi) * (phi - 1) < 1 * (2 - 1) := by
            apply mul_lt_mul'
            · exact h_frac.le
            · linarith
            · linarith
            · exact h_frac
          linarith
        · -- fract * phi ? No.
          -- Let's re-evaluate.
          sorry
    sorry
  · intro n
    unfold imo_f
    apply Int.toNat_lt_toNat
    · apply (tsub_le_tsub_right).mpr
      apply Int.floor_le
      positivity
    · apply Int.lt_floor_add_one_of_le
      rw [Int.cast_sub, Int.cast_one, sub_add_cancel]
      apply (add_le_add_right).mpr
      have : ((n + 1 : ℕ) : ℝ) * phi + phi = ((n + 1 + 1 : ℕ) : ℝ) * phi := by ring
      rw [← this]
      have hphi : 1 < phi := by
        unfold phi
        apply (lt_div_iff₀ (by norm_num)).mpr
        apply (lt_sub_iff_add_lt).mpr
        rw [show (1 : ℝ) * 2 - 1 = 1 by norm_num]
        apply (lt_sqrt (by norm_num) (by norm_num)).mpr
        norm_num
      apply (le_add_of_le_of_one_le (le_refl _) hphi.le)

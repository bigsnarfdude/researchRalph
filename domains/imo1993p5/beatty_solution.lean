import Mathlib.NumberTheory.Rayleigh
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Real.Basic

open Real Nat

noncomputable section

def phi : ℝ := (1 + sqrt 5) / 2

lemma phi_pos : 0 < phi := by
  unfold phi
  have : 0 < sqrt 5 := sqrt_pos.mpr (by norm_num)
  positivity

lemma phi_gt_one : 1 < phi := by
  unfold phi
  rw [lt_div_iff (by norm_num)]
  have : 1 < sqrt 5 := by
    rw [← sqrt_one, sqrt_lt_sqrt_iff (by norm_num)]
    norm_num
  linarith

lemma phi_sq : phi^2 = phi + 1 := by
  unfold phi
  have h5 : (sqrt 5)^2 = 5 := sq_sqrt (by norm_num)
  ring_nf
  rw [h5]
  ring

lemma phi_inv_add_phi_sq_inv : 1/phi + 1/phi^2 = 1 := by
  have hphi : phi ≠ 0 := phi_pos.ne'
  have hphi2 : phi^2 ≠ 0 := by positivity
  field_simp
  rw [phi_sq]
  ring

lemma phi_hc : Real.HolderConjugate phi (phi^2) := by
  constructor
  · exact phi_gt_one
  · rw [phi_sq]; linarith [phi_pos]
  · exact phi_inv_add_phi_sq_inv

def a (n : ℤ) : ℤ := ⌊(n : ℝ) * phi⌋
def b (n : ℤ) : ℤ := ⌊(n : ℝ) * phi^2⌋

lemma b_eq_a_add (n : ℤ) : b n = a n + n := by
  unfold a b
  rw [phi_sq, Int.cast_add, Int.cast_one, mul_add, mul_one, Int.floor_add_int]

lemma a_pos {n : ℤ} (hn : 0 < n) : 0 < a n := by
  unfold a
  apply Int.floor_pos.mpr
  apply mul_pos (Int.cast_pos.mpr hn) phi_pos

lemma a_a_eq_b_sub_one {n : ℤ} (hn : 0 < n) : a (a n) = b n - 1 := by
  unfold a b
  rw [phi_sq, Int.cast_add, Int.cast_one, mul_add, mul_one]
  have h_floor := Int.floor_eq_iff.mpr ⟨le_refl (a n : ℝ), Int.lt_floor_add_one ((n : ℝ) * phi)⟩
  have h_fract := Int.fract_pos.mpr (phi_irrational_sqrt 5 (by norm_num)).rat_mul_irrational.rat_add_irrational.rat_mul_irrational.ne_int
  -- Wait, I don't need to be so complex.
  -- (a n) * phi = (n * phi - {n * phi}) * phi = n * phi^2 - {n * phi} * phi
  -- = n * phi + n - {n * phi} * phi = a n + {n * phi} + n - {n * phi} * phi
  -- = a n + n - {n * phi} * (phi - 1)
  -- Since 0 < {n * phi} < 1 and 0 < phi - 1 < 1, their product is in (0, 1).
  -- So a n + n - delta has floor a n + n - 1.
  -- And b n = a n + n.
  set fract := Int.fract ((n : ℝ) * phi)
  have h_eq : (a n : ℝ) = (n : ℝ) * phi - fract := by
    unfold a; rw [Int.fract]; ring
  have h_fract_pos : 0 < fract := by
    rw [Int.fract_pos]
    apply Irrational.ne_int
    apply Irrational.rat_mul_irrational
    · exact_mod_cast hn.ne'
    · unfold phi
      apply Irrational.rat_add_irrational (by norm_num)
      apply Irrational.rat_mul_irrational (by norm_num)
      exact irrational_sqrt_five
  have h_fract_lt_one : fract < 1 := Int.fract_lt_one _
  have h_phi_lt_2 : phi < 2 := by
    unfold phi
    rw [div_lt_iff (by norm_num)]
    apply (lt_sub_iff_add_lt).mpr
    rw [show (2 : ℝ) * 2 - 1 = 3 by norm_num]
    rw [← sq_lt_sq (by positivity) (by norm_num), sq_sqrt (by norm_num)]
    norm_num
  have h_phi_gt_1 : 1 < phi := phi_gt_one
  
  have h_val : (a n : ℝ) * phi = (a n + n : ℝ) - fract * (phi - 1) := by
    rw [h_eq, sub_mul, phi_sq, add_mul, one_mul]
    have : (n : ℝ) * phi = (a n : ℝ) + fract := by rw [h_eq]; ring
    rw [this]
    ring
  
  apply Int.floor_eq_iff.mpr
  constructor
  · -- a n + n - 1 <= a n + n - fract * (phi - 1)
    -- fract * (phi - 1) <= 1
    rw [le_sub_iff_add_le, add_comm, ← le_sub_iff_add_le]
    have : fract * (phi - 1) < 1 * 1 := by
      apply mul_lt_mul'
      · exact h_fract_lt_one.le
      · linarith
      · linarith
      · exact h_fract_lt_one
    rw [mul_one] at this
    linarith
  · -- a n + n - fract * (phi - 1) < a n + n
    -- 0 < fract * (phi - 1)
    rw [sub_lt_self_iff]
    apply mul_pos h_fract_pos
    linarith

def imo_f (n : ℕ) : ℕ :=
  Int.toNat (a (n + 1) - 1)

lemma imo_f_val (n : ℕ) : (imo_f n : ℤ) = a (n + 1) - 1 := by
  unfold imo_f
  apply Int.toNat_of_nonneg
  have h1 : 1 ≤ n + 1 := by omega
  have h2 : 0 < a (n + 1) := a_pos (by exact_mod_cast h1)
  omega

theorem imo_1993_p5_beatty :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ (∀ n, f (f n) = f n + n) ∧ ∀ n, f n < f (n + 1) := by
  use imo_f
  constructor
  · -- f 1 = 2
    rw [← Int.cast_inj (R := ℤ), imo_f_val]
    unfold a phi
    have h1 : 3 ≤ (2 : ℝ) * ((1 + sqrt 5) / 2) := by
      rw [mul_div_cancel₀ _ (by norm_num)]
      apply (le_sub_iff_add_le).mpr
      rw [show (3 : ℝ) - 1 = 2 by norm_num]
      apply (le_sqrt (by norm_num) (by norm_num)).mpr
      norm_num
    have h2 : (2 : ℝ) * ((1 + sqrt 5) / 2) < 4 := by
      rw [mul_div_cancel₀ _ (by norm_num)]
      apply (lt_sub_iff_add_lt).mpr
      rw [show (4 : ℝ) - 1 = 3 by norm_num]
      apply (sqrt_lt (by norm_num) (by norm_num)).mpr
      norm_num
    have : ⌊(2 : ℝ) * phi⌋ = 3 := Int.floor_eq_iff.mpr ⟨h1, h2⟩
    rw [this]
    norm_num
  constructor
  · intro n
    rw [← Int.cast_inj (R := ℤ), imo_f_val]
    set m := imo_f n
    have hm : (m : ℤ) = a (n + 1) - 1 := imo_f_val n
    rw [imo_f_val]
    -- Goal: a (m + 1) - 1 = (a (n + 1) - 1) + n
    -- a (a (n + 1) - 1 + 1) - 1 = a (n + 1) - 1 + n
    -- a (a (n + 1)) = a (n + 1) + n
    -- a (a (n + 1)) = b (n + 1) - 1
    rw [show (m : ℤ) + 1 = a (n + 1) by rw [hm]; omega]
    rw [a_a_eq_b_sub_one (by exact_mod_cast succ_pos n)]
    rw [b_eq_a_add]
    omega
  · intro n
    rw [← Int.cast_lt (α := ℤ), imo_f_val, imo_f_val]
    apply Int.sub_lt_sub_right
    unfold a
    apply Int.floor_mono
    apply mul_le_mul_of_nonneg_right
    · norm_cast; linarith
    · exact phi_pos.le

theorem imo_1993_p5 :
    ∃ f : ℕ → ℕ, f 1 = 2 ∧ ∀ n, f (f n) = f n + n ∧ ∀ n, f n < f (n + 1) :=
  imo_1993_p5_beatty


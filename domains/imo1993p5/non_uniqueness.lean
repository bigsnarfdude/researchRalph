import Mathlib

set_option maxHeartbeats 0

def greatestFib (n : ℕ) : ℕ := 
  (List.range (n + 3)).filter (λ k => fib k ≤ n) |>.maximum |>.getD 0

def f_zeck : ℕ → ℕ
  | 0 => 0
  | n + 1 =>
    let k := greatestFib (n + 1)
    fib (k + 1) + f_zeck ((n + 1) - fib k)
termination_by n => n

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def f_floor (n : ℕ) : ℕ :=
  Int.toNat ⌊(n : ℝ) * phi + 0.5⌋

theorem zeck_4 : f_zeck 4 = 7 := by
  native_decide

theorem floor_4 : f_floor 4 = 6 := by
  -- We can't use native_decide for noncomputable Real
  unfold f_floor phi
  have h1 : 6 ≤ 4 * ((1 + Real.sqrt 5) / 2) + 0.5 := by
    apply (le_sub_iff_add_le).mpr
    rw [show (6 : ℝ) - 0.5 = 5.5 by norm_num]
    apply (le_div_iff₀ (by norm_num)).mpr
    rw [show (5.5 : ℝ) * 2 = 11 by norm_num]
    apply (le_sub_iff_add_le).mpr
    rw [show (11 : ℝ) - 4 = 7 by norm_num]
    apply (Real.le_sqrt (by norm_num) (by norm_num)).mpr
    norm_num
  have h2 : 4 * ((1 + Real.sqrt 5) / 2) + 0.5 < 7 := by
    apply (lt_sub_iff_add_lt).mpr
    rw [show (7 : ℝ) - 0.5 = 6.5 by norm_num]
    apply (div_lt_iff₀ (by norm_num)).mpr
    rw [show (6.5 : ℝ) * 2 = 13 by norm_num]
    apply (lt_sub_iff_add_lt).mpr
    rw [show (13 : ℝ) - 4 = 9 by norm_num]
    apply (Real.sqrt_lt (by norm_num) (by norm_num)).mpr
    norm_num
  rw [Int.toNat_eq_iff (by norm_num)]
  apply Int.floor_eq_iff.mpr
  exact ⟨h1, h2⟩

theorem non_uniqueness_at_4 : f_zeck 4 ≠ f_floor 4 := by
  rw [zeck_4, floor_4]
  norm_num

import Mathlib
set_option maxHeartbeats 128000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
open BigOperators Complex Finset

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  have hz4 : z ^ 4 = -1 := by
    rw [h₀, div_pow]
    have h1 : ((1:ℂ) + I) ^ 4 = -4 := by
      have : (1 + I : ℂ) ^ 2 = 2 * I := by
        have := I_sq; rw [show (1 + I : ℂ) ^ 2 = 1 + 2 * I + I ^ 2 from by ring, I_sq]; ring
      calc (1 + I : ℂ) ^ 4 = ((1 + I) ^ 2) ^ 2 := by ring
        _ = (2 * I) ^ 2 := by rw [this]
        _ = -4 := by rw [show (2 * I) ^ 2 = 4 * I ^ 2 from by ring, I_sq]; ring
    have h3 : (↑(Real.sqrt 2) : ℂ) ^ 4 = 4 := by
      exact_mod_cast show Real.sqrt 2 ^ 4 = 4 from by
        nlinarith [Real.sq_sqrt (show (2:ℝ) ≥ 0 from by norm_num)]
    rw [h1, h3]; ring
  have hz8 : z ^ 8 = 1 := by
    have : z ^ 8 = (z ^ 4) ^ 2 := by ring
    rw [this, hz4]; ring
  have hz_ne : z ≠ 0 := by
    rw [h₀]; apply div_ne_zero
    · intro h; have := congr_arg Complex.re h; simp at this
    · exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num : (0:ℝ) < 2)
  have hmod (k : ℕ) : z ^ (k ^ 2) = z ^ (k ^ 2 % 8) := by
    conv_lhs => rw [show k^2 = 8*(k^2/8) + k^2%8 from (Nat.div_add_mod _ 8).symm]
    rw [pow_add, pow_mul, hz8, one_pow, one_mul]
  have hS1 : ∑ k ∈ Icc 1 12, z ^ k ^ 2 = 6 * z := by
    rw [show Icc (1:ℕ) 12 = {1,2,3,4,5,6,7,8,9,10,11,12} from by decide]
    simp [Finset.sum_insert, Finset.sum_singleton, Finset.mem_insert, Finset.mem_singleton, hmod]
    simp only [pow_zero, pow_one, hz4]
    ring
  have hS2 : ∑ k ∈ Icc 1 12, 1 / z ^ k ^ 2 = 6 / z := by
    rw [show Icc (1:ℕ) 12 = {1,2,3,4,5,6,7,8,9,10,11,12} from by decide]
    simp [Finset.sum_insert, Finset.sum_singleton, Finset.mem_insert, Finset.mem_singleton, hmod]
    simp only [pow_zero, pow_one, hz4]
    field_simp
    ring
  rw [hS1, hS2]; field_simp; ring

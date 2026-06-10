import Mathlib
set_option maxHeartbeats 64000000
set_option maxRecDepth 2000
set_option linter.all false
open BigOperators Complex

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + I) / ↑(Real.sqrt 2)) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  have hz4 : z ^ 4 = -1 := by
    have hz2 : z ^ 2 = I := by
      rw [h₀]; apply Complex.ext <;> simp [sq] <;> field_simp <;> ring_nf <;>
      rw [Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)]
    have h : z ^ 4 = (z ^ 2) ^ 2 := by ring
    rw [h, hz2, I_sq]
  have hz8 : z ^ 8 = 1 := by
    have h : z ^ 8 = (z ^ 4) ^ 2 := by ring
    rw [h, hz4]; ring
  have hzne : z ≠ 0 := by
    rw [h₀]; apply div_ne_zero
    · intro h; have := congr_arg Complex.re h; simp at this
    · exact_mod_cast Real.sqrt_ne_zero'.mpr (by norm_num : (0:ℝ) < 2)
  -- Rewrite everything in terms of z and z^4=-1 and z^8=1
  have hp : ∀ n : ℕ, z ^ (8 * n) = 1 := fun n => by rw [pow_mul, hz8, one_pow]
  -- All k² mod 8: 1→1, 4→4, 9→1, 16→0, 25→1, 36→4, 49→1, 64→0, 81→1, 100→4, 121→1, 144→0
  have rp : z^9 = z ∧ z^16 = 1 ∧ z^25 = z ∧ z^36 = z^4 ∧ z^49 = z ∧ z^64 = 1 ∧
    z^81 = z ∧ z^100 = z^4 ∧ z^121 = z ∧ z^144 = 1 := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    · try (rw [show (9:ℕ) = 8*1+1 from rfl, pow_add, hp, one_mul, pow_one])
      try (rw [show (16:ℕ) = 8*2 from rfl]; exact hp 2)
      try (rw [show (25:ℕ) = 8*3+1 from rfl, pow_add, hp, one_mul, pow_one])
      try (rw [show (36:ℕ) = 8*4+4 from rfl, pow_add, hp, one_mul])
      try (rw [show (49:ℕ) = 8*6+1 from rfl, pow_add, hp, one_mul, pow_one])
      try (rw [show (64:ℕ) = 8*8 from rfl]; exact hp 8)
      try (rw [show (81:ℕ) = 8*10+1 from rfl, pow_add, hp, one_mul, pow_one])
      try (rw [show (100:ℕ) = 8*12+4 from rfl, pow_add, hp, one_mul])
      try (rw [show (121:ℕ) = 8*15+1 from rfl, pow_add, hp, one_mul, pow_one])
      try (rw [show (144:ℕ) = 8*18 from rfl]; exact hp 18)
  obtain ⟨r9, r16, r25, r36, r49, r64, r81, r100, r121, r144⟩ := rp
  -- Expand sums
  rw [show Finset.Icc (1:ℕ) 12 = ({1,2,3,4,5,6,7,8,9,10,11,12} : Finset ℕ) from by native_decide]
  simp (config := { decide := true }) only [Finset.sum_insert, Finset.sum_singleton,
    Finset.mem_insert, Finset.mem_singleton]
  norm_num
  -- Rewrite powers in both sums
  rw [r9, r16, r25, r36, r49, r64, r81, r100, r121, r144, hz4]
  -- Now: (z + -1 + z + 1 + z + -1 + z + 1 + z + -1 + z + 1) *
  -- (z⁻¹ + (-1)⁻¹ + z⁻¹ + 1⁻¹ + z⁻¹ + (-1)⁻¹ + z⁻¹ + 1⁻¹ + z⁻¹ + (-1)⁻¹ + z⁻¹ + 1⁻¹)
  -- = 6z * 6z⁻¹ = 36
  field_simp
  ring

import Mathlib
set_option maxHeartbeats 32000000

open BigOperators Real Nat Topology Rat Complex

theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :
  ((∑ k ∈ Finset.Icc 1 12, z ^ k ^ 2) * (∑ k ∈ Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by
  -- Key: z = e^{iπ/4}, so z^2 = i, z^4 = -1, z^8 = 1
  have hz2 : z ^ 2 = Complex.I := by
    rw [h₀]; field_simp; ring_nf
    rw [Real.sq_sqrt (by norm_num : (2:ℝ) ≥ 0)]
    ring
  have hz4 : z ^ 4 = -1 := by
    have : z ^ 4 = (z ^ 2) ^ 2 := by ring
    rw [this, hz2]; simp [Complex.I_sq]
  have hz8 : z ^ 8 = 1 := by
    have : z ^ 8 = (z ^ 4) ^ 2 := by ring
    rw [this, hz4]; ring
  -- z^(k² mod 8) for k=1..12:
  -- k=1: 1→z, k=2: 4→z⁴=-1, k=3: 1→z, k=4: 0→1, k=5: 1→z, k=6: 4→-1
  -- k=7: 1→z, k=8: 0→1, k=9: 1→z, k=10: 4→-1, k=11: 1→z, k=12: 0→1
  -- Sum = 6z + 3(-1) + 3(1) = 6z
  -- Similarly ∑1/z^(k²) = 6/z + 3(-1) + 3(1) = 6/z
  -- Product = 36
  sorry

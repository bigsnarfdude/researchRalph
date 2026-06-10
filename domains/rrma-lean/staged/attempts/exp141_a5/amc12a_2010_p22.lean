import Mathlib
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  have hsplit : Finset.Icc (1:ℤ) 119 = Finset.Icc 1 84 ∪ Finset.Icc 85 119 := by
    ext k; simp only [Finset.mem_union, Finset.mem_Icc]; omega
  have hdisj : Disjoint (Finset.Icc (1:ℤ) 84) (Finset.Icc 85 119) := by
    apply Finset.disjoint_left.mpr
    intro k hk1 hk2; simp only [Finset.mem_Icc] at hk1 hk2; omega
  rw [hsplit, Finset.sum_union hdisj]
  have tri1 : |∑ k ∈ Finset.Icc (1:ℤ) 84, ((k : ℝ) * x - 1)| ≤
              ∑ k ∈ Finset.Icc (1:ℤ) 84, |((k : ℝ) * x - 1)| := by
    have h := norm_sum_le (Finset.Icc (1:ℤ) 84) (fun k => (k : ℝ) * x - 1)
    simp only [Real.norm_eq_abs] at h; exact h
  have tri2 : |∑ k ∈ Finset.Icc (85:ℤ) 119, ((k : ℝ) * x - 1)| ≤
              ∑ k ∈ Finset.Icc (85:ℤ) 119, |((k : ℝ) * x - 1)| := by
    have h := norm_sum_le (Finset.Icc (85:ℤ) 119) (fun k => (k : ℝ) * x - 1)
    simp only [Real.norm_eq_abs] at h; exact h
  have hs1 : ∑ k ∈ Finset.Icc (1:ℤ) 84, ((k : ℝ) * x - 1) = 3570 * x - 84 := by
    simp_rw [show ∀ k : ℤ, (k : ℝ) * x - 1 = x * (k : ℝ) + (-1 : ℝ) from fun k => by ring]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul]
    have hk : ∑ k ∈ Finset.Icc (1:ℤ) 84, (k : ℝ) = 3570 := by
      have h1 : ∑ k ∈ Finset.Icc (1:ℤ) 84, k = (3570 : ℤ) := by native_decide
      have h2 : (∑ k ∈ Finset.Icc (1:ℤ) 84, (k : ℝ)) = ((∑ k ∈ Finset.Icc (1:ℤ) 84, k : ℤ) : ℝ) := by
        simp only [Int.cast_sum]
      rw [h2, h1]; norm_num
    have hc : ((Finset.Icc (1:ℤ) 84).card : ℝ) = 84 := by
      have h : (Finset.Icc (1:ℤ) 84).card = 84 := by native_decide
      simp [h]
    rw [hk, hc]; ring
  have hs2 : ∑ k ∈ Finset.Icc (85:ℤ) 119, ((k : ℝ) * x - 1) = 3570 * x - 35 := by
    simp_rw [show ∀ k : ℤ, (k : ℝ) * x - 1 = x * (k : ℝ) + (-1 : ℝ) from fun k => by ring]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul]
    have hk : ∑ k ∈ Finset.Icc (85:ℤ) 119, (k : ℝ) = 3570 := by
      have h1 : ∑ k ∈ Finset.Icc (85:ℤ) 119, k = (3570 : ℤ) := by native_decide
      have h2 : (∑ k ∈ Finset.Icc (85:ℤ) 119, (k : ℝ)) = ((∑ k ∈ Finset.Icc (85:ℤ) 119, k : ℤ) : ℝ) := by
        simp only [Int.cast_sum]
      rw [h2, h1]; norm_num
    have hc : ((Finset.Icc (85:ℤ) 119).card : ℝ) = 35 := by
      have h : (Finset.Icc (85:ℤ) 119).card = 35 := by native_decide
      simp [h]
    rw [hk, hc]; ring
  have hrev : (49 : ℝ) ≤ |3570 * x - 84| + |3570 * x - 35| := by
    have key := norm_add_le (3570 * x - 84) (35 - 3570 * x)
    simp only [Real.norm_eq_abs] at key
    have h1 : (3570 * x - 84) + (35 - 3570 * x) = -49 := by ring
    rw [h1, abs_neg] at key
    rw [abs_sub_comm (35 : ℝ) (3570 * x)] at key
    norm_num at key
    linarith
  calc (49 : ℝ)
      ≤ |3570 * x - 84| + |3570 * x - 35| := hrev
    _ = |∑ k ∈ Finset.Icc (1:ℤ) 84, ((k : ℝ) * x - 1)| +
        |∑ k ∈ Finset.Icc (85:ℤ) 119, ((k : ℝ) * x - 1)| := by rw [hs1, hs2]
    _ ≤ (∑ k ∈ Finset.Icc (1:ℤ) 84, |((k : ℝ) * x - 1)|) +
        (∑ k ∈ Finset.Icc (85:ℤ) 119, |((k : ℝ) * x - 1)|) := add_le_add tri1 tri2

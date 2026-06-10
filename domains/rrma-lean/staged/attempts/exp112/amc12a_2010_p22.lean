import Mathlib
set_option maxHeartbeats 64000000
set_option maxRecDepth 4096
open BigOperators Real Nat Topology Rat Finset

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  have hAB : Finset.Icc (1:ℤ) 119 = Finset.Icc 1 84 ∪ Finset.Icc 85 119 := by
    ext k; simp only [Finset.mem_union, Finset.mem_Icc]; omega
  have hDisj : Disjoint (Finset.Icc (1:ℤ) 84) (Finset.Icc 85 119) := by
    simp only [Finset.disjoint_left, Finset.mem_Icc]; omega
  rw [hAB, Finset.sum_union hDisj]
  have key : (∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1)) - 
             (∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1)) = -49 := by
    simp only [Finset.sum_sub_distrib, ← Finset.sum_mul, Finset.sum_const]
    have h3 : (Finset.Icc (1:ℤ) 84).card = 84 := by native_decide
    have h4 : (Finset.Icc (85:ℤ) 119).card = 35 := by native_decide
    rw [h3, h4]
    rw [show (∑ k ∈ Finset.Icc (1:ℤ) 84, (↑k : ℝ)) = 
            ((∑ k ∈ Finset.Icc (1:ℤ) 84, k : ℤ) : ℝ) from by push_cast; rfl]
    rw [show (∑ k ∈ Finset.Icc (85:ℤ) 119, (↑k : ℝ)) = 
            ((∑ k ∈ Finset.Icc (85:ℤ) 119, k : ℤ) : ℝ) from by push_cast; rfl]
    have h1 : (∑ k ∈ Finset.Icc (1:ℤ) 84, k) = 3570 := by native_decide
    have h2 : (∑ k ∈ Finset.Icc (85:ℤ) 119, k) = 3570 := by native_decide
    rw [h1, h2]; push_cast; ring
  calc (49:ℝ) 
      ≤ |∑ k ∈ Finset.Icc (1:ℤ) 84, (↑k * x - 1)| + 
        |∑ k ∈ Finset.Icc (85:ℤ) 119, (↑k * x - 1)| := by
        have h49 : (49:ℝ) = |(-49:ℝ)| := by norm_num
        rw [h49, ← key]
        exact abs_sub _ _
    _ ≤ (∑ k ∈ Finset.Icc (1:ℤ) 84, |↑k * x - 1|) + 
        (∑ k ∈ Finset.Icc (85:ℤ) 119, |↑k * x - 1|) := by
        gcongr
        · exact Finset.abs_sum_le_sum_abs _ _
        · exact Finset.abs_sum_le_sum_abs _ _

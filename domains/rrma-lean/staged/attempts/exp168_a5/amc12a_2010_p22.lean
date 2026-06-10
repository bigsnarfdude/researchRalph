import Mathlib
set_option maxHeartbeats 8000000
set_option maxRecDepth 2000
open BigOperators Real Nat Topology Rat

private lemma cast_sum_ℤ_ℝ (a b : ℤ) (v : ℤ) (h : ∑ k ∈ Finset.Icc a b, k = v) :
    ∑ k ∈ Finset.Icc a b, (k : ℝ) = (v : ℝ) := by
  have hcast : (↑(∑ k ∈ Finset.Icc a b, k) : ℝ) = ∑ k ∈ Finset.Icc a b, (↑k : ℝ) := by
    simp only [Int.cast_sum]
  linarith [hcast, show (↑(∑ k ∈ Finset.Icc a b, k) : ℝ) = (↑v : ℝ) from by exact_mod_cast h]

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  have hsplit : Finset.Icc (1:ℤ) 119 = Finset.Icc 1 84 ∪ Finset.Icc 85 119 := by
    ext k; simp only [Finset.mem_Icc, Finset.mem_union]; omega
  have hdisj : Disjoint (Finset.Icc (1:ℤ) 84) (Finset.Icc 85 119) := by
    rw [Finset.disjoint_left]; intro k hk1 hk2; simp [Finset.mem_Icc] at *; omega
  rw [hsplit, Finset.sum_union hdisj]
  have tri1 : |∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1)| ≤
    ∑ k ∈ Finset.Icc (1:ℤ) 84, |↑k * x - 1| := Finset.abs_sum_le_sum_abs _ _
  have tri2 : |∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1)| ≤
    ∑ k ∈ Finset.Icc (85:ℤ) 119, |↑k * x - 1| := Finset.abs_sum_le_sum_abs _ _
  have hksum1 : ∑ k ∈ Finset.Icc (1:ℤ) 84, (k : ℝ) = 3570 :=
    cast_sum_ℤ_ℝ 1 84 3570 (by native_decide)
  have hksum2 : ∑ k ∈ Finset.Icc (85:ℤ) 119, (k : ℝ) = 3570 :=
    cast_sum_ℤ_ℝ 85 119 3570 (by native_decide)
  have hcard1 : (Finset.Icc (1:ℤ) 84).card = 84 := by decide
  have hcard2 : (Finset.Icc (85:ℤ) 119).card = 35 := by decide
  have hS1 : ∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1) = 3570 * x - 84 := by
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hksum1, Finset.sum_const, hcard1]; ring
  have hS2 : ∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1) = 3570 * x - 35 := by
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hksum2, Finset.sum_const, hcard2]; ring
  have h_ab : |∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1) -
    ∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1)| = 49 := by
    rw [hS1, hS2]; ring_nf; simp
  have abs_tri : |∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1) -
    ∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1)| ≤
    |∑ k ∈ Finset.Icc (1:ℤ) 84, ((↑k : ℝ) * x - 1)| +
    |∑ k ∈ Finset.Icc (85:ℤ) 119, ((↑k : ℝ) * x - 1)| := abs_sub _ _
  linarith

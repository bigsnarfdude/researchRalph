import Mathlib
set_option maxHeartbeats 128000000
open BigOperators Real Nat Topology Rat

theorem aime_1991_p6 (r : ℝ) (h₀ : (∑ k ∈ Finset.Icc (19 : ℕ) 91, Int.floor (r + k / 100)) = 546) :
  Int.floor (100 * r) = 743 := by
  set n := ⌊r⌋; set f := r - ↑n
  have hf0 : 0 ≤ f := sub_nonneg.mpr (Int.floor_le r)
  have hf1 : f < 1 := by linarith [Int.lt_floor_add_one r]
  have hterm : ∀ k ∈ Finset.Icc (19:ℕ) 91,
    ⌊r + ↑k / 100⌋ = n + ⌊f + ↑k / 100⌋ := by
    intro k _; rw [show r + ↑k / 100 = ↑n + (f + ↑k / 100) from by simp [f]; ring]
    exact Int.floor_intCast_add n _
  rw [Finset.sum_congr rfl hterm] at h₀
  simp only [Finset.sum_add_distrib, Finset.sum_const,
    show (Finset.Icc (19:ℕ) 91).card = 73 from by native_decide, nsmul_eq_mul] at h₀
  -- h₀ : ↑73 * n + S = 546
  set S := ∑ k ∈ Finset.Icc (19:ℕ) 91, ⌊f + ↑k / 100⌋ with hS_def
  have hfl_nn : ∀ k : ℕ, 0 ≤ ⌊f + ↑k / 100⌋ := fun k => Int.floor_nonneg.mpr (by positivity)
  have hfl_le1 : ∀ k ∈ Finset.Icc (19:ℕ) 91, ⌊f + ↑k / 100⌋ ≤ 1 := by
    intro k hk; have : (k:ℝ) ≤ 91 := by exact_mod_cast (Finset.mem_Icc.mp hk).2
    have := Int.floor_lt.mpr (show f + ↑k / 100 < (2:ℤ) from by push_cast; linarith); omega
  have hS_lb : 0 ≤ S := Finset.sum_nonneg (fun k _ => hfl_nn k)
  have hS_ub : S ≤ 73 := by
    calc S ≤ ∑ _ ∈ Finset.Icc (19:ℕ) 91, (1:ℤ) := Finset.sum_le_sum hfl_le1
      _ = 73 := by simp [show (Finset.Icc (19:ℕ) 91).card = 73 from by native_decide]
  -- h₀ has ↑73 which is (73:ℕ)→ℤ. Normalize.
  have h₀' : 73 * n + S = 546 := by exact_mod_cast h₀
  have hn7 : n = 7 := by omega
  have hS35 : S = 35 := by omega
  rw [show 100 * r = ↑(700:ℤ) + 100 * f from by simp [f]; rw [hn7]; push_cast; ring,
      Int.floor_intCast_add]
  suffices h : 43 ≤ 100 * f ∧ 100 * f < 44 by
    have := Int.le_floor.mpr (show (43:ℤ) ≤ 100*f from by push_cast; linarith)
    have := Int.floor_lt.mpr (show 100*f < (44:ℤ) from by push_cast; linarith); omega
  -- Split {19,...,91} = {19,...,57} ∪ {58,...,91} (and also {19,...,55} ∪ {56,...,91})
  have hsp1 : Finset.Icc (19:ℕ) 91 = Finset.Icc 19 57 ∪ Finset.Icc 58 91 := by
    ext k; simp only [Finset.mem_union, Finset.mem_Icc]; omega
  have hdj1 : Disjoint (Finset.Icc (19:ℕ) 57) (Finset.Icc 58 91) := by
    simp only [Finset.disjoint_left, Finset.mem_Icc]; omega
  have hsp2 : Finset.Icc (19:ℕ) 91 = Finset.Icc 19 55 ∪ Finset.Icc 56 91 := by
    ext k; simp only [Finset.mem_union, Finset.mem_Icc]; omega
  have hdj2 : Disjoint (Finset.Icc (19:ℕ) 55) (Finset.Icc 56 91) := by
    simp only [Finset.disjoint_left, Finset.mem_Icc]; omega
  constructor
  · -- 43 ≤ 100f
    by_contra h; push_neg at h
    have : S ≤ 34 := by
      rw [hS_def, hsp1, Finset.sum_union hdj1]
      have : ∑ k ∈ Finset.Icc (19:ℕ) 57, ⌊f + ↑k / 100⌋ = 0 :=
        Finset.sum_eq_zero fun k hk => Int.floor_eq_zero_iff.mpr
          ⟨by positivity, by linarith [show (k:ℝ) ≤ 57 from by exact_mod_cast (Finset.mem_Icc.mp hk).2]⟩
      have : ∑ k ∈ Finset.Icc (58:ℕ) 91, ⌊f + ↑k / 100⌋ ≤ 34 := by
        calc _ ≤ ∑ _ ∈ Finset.Icc (58:ℕ) 91, (1:ℤ) :=
              Finset.sum_le_sum fun k hk => hfl_le1 k (by simp [Finset.mem_Icc] at hk ⊢; omega)
          _ = 34 := by simp [show (Finset.Icc (58:ℕ) 91).card = 34 from by native_decide]
      linarith
    omega
  · -- 100f < 44
    by_contra h; push_neg at h
    have : 36 ≤ S := by
      rw [hS_def, hsp2, Finset.sum_union hdj2]
      have : 36 ≤ ∑ k ∈ Finset.Icc (56:ℕ) 91, ⌊f + ↑k / 100⌋ := by
        calc 36 = ∑ _ ∈ Finset.Icc (56:ℕ) 91, (1:ℤ) := by
              simp [show (Finset.Icc (56:ℕ) 91).card = 36 from by native_decide]
          _ ≤ _ := Finset.sum_le_sum fun k hk => Int.le_floor.mpr
              (by push_cast; linarith [show (56:ℝ) ≤ (k:ℝ) from by exact_mod_cast (Finset.mem_Icc.mp hk).1])
      linarith [Finset.sum_nonneg (fun (k : ℕ) (_ : k ∈ Finset.Icc 19 55) => hfl_nn k)]
    omega

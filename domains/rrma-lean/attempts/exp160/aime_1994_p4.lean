import Mathlib
set_option maxHeartbeats 12800000
set_option maxRecDepth 4096
open BigOperators Real Nat Topology Rat

lemma floor_logb_eq_nat_log (k : ℕ) (hk : 1 ≤ k) : 
    ⌊Real.logb 2 ↑k⌋ = ↑(Nat.log 2 k) := by
  set m := Nat.log 2 k
  have hk_ne : k ≠ 0 := by omega
  have hk_pos : (0 : ℝ) < ↑k := by exact_mod_cast show 0 < k by omega
  have hlo_r : (↑m : ℤ) ≤ ⌊Real.logb 2 ↑k⌋ := by
    rw [Int.le_floor]; push_cast
    rw [Real.le_logb_iff_rpow_le (by norm_num : (1:ℝ) < 2) hk_pos, Real.rpow_natCast]
    exact_mod_cast Nat.pow_log_le_self 2 hk_ne
  have hhi_r : ⌊Real.logb 2 ↑k⌋ < (↑m : ℤ) + 1 := by
    rw [Int.floor_lt]; push_cast
    rw [Real.logb_lt_iff_lt_rpow (by norm_num : (1:ℝ) < 2) hk_pos]
    rw [show (↑m : ℝ) + 1 = ↑(m + 1 : ℕ) from by push_cast; ring, Real.rpow_natCast]
    exact_mod_cast Nat.lt_pow_succ_log_self (by omega : 1 < 2) k
  omega

theorem aime_1994_p4 (n : ℕ) (h₀ : 0 < n)
  (h₀' : (∑ k ∈ Finset.Icc 1 n, Int.floor (Real.logb 2 k)) = 1994) : n = 312 := by
  have h₁ : (∑ k ∈ Finset.Icc 1 n, (↑(Nat.log 2 k) : ℤ)) = 1994 := by
    convert h₀' using 1; apply Finset.sum_congr rfl
    intro k hk; rw [Finset.mem_Icc] at hk; exact (floor_logb_eq_nat_log k hk.1).symm
  have h₂ : (∑ k ∈ Finset.Icc 1 n, Nat.log 2 k : ℕ) = 1994 := by exact_mod_cast h₁
  have hf312 : (∑ k ∈ Finset.Icc 1 312, Nat.log 2 k) = 1994 := by native_decide
  have hf311 : (∑ k ∈ Finset.Icc 1 311, Nat.log 2 k) = 1986 := by native_decide
  by_contra hne; rcases lt_or_gt_of_ne hne with h | h
  · -- n ≤ 311
    have hsub : Finset.Icc 1 n ⊆ Finset.Icc 1 311 := Finset.Icc_subset_Icc_right (by omega)
    have : ∑ k ∈ Finset.Icc 1 n, Nat.log 2 k ≤ ∑ k ∈ Finset.Icc 1 311, Nat.log 2 k :=
      Finset.sum_le_sum_of_subset_of_nonneg hsub (fun _ _ _ => Nat.zero_le _)
    omega
  · -- n ≥ 313
    have hsplit : Finset.Icc (1:ℕ) n = Finset.Icc 1 312 ∪ Finset.Icc 313 n := by
      ext k; simp only [Finset.mem_Icc, Finset.mem_union]; omega
    have hdisj : Disjoint (Finset.Icc (1:ℕ) 312) (Finset.Icc 313 n) := by
      rw [Finset.disjoint_left]; intro k; simp only [Finset.mem_Icc]; omega
    rw [hsplit, Finset.sum_union hdisj] at h₂
    have hpos : 0 < ∑ k ∈ Finset.Icc 313 n, Nat.log 2 k :=
      Finset.sum_pos (fun k hk => by simp [Finset.mem_Icc] at hk; exact Nat.log_pos (by omega) (by omega))
        ⟨313, by simp [Finset.mem_Icc]; omega⟩
    omega

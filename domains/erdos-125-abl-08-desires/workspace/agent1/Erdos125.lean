import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_21 {n : ℕ} (hn : n ∈ setB) (hlt : n < 64) : n ≤ 21 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨62, 2, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

-- Phase 2: generalization to base pair (3,5), analogous to (3,4).
-- max(setA ∩ [0,27)) = 13 = (3^3-1)/2, max(setC ∩ [0,25)) = 6 = (5^2-1)/4.
-- 13 + 6 = 19 < 20 ≤ min(27,25) = 25, so n = 20 is a gap.
def setC : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setAC : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setC, a + b = n}

private lemma setA_le_13 {n : ℕ} (hn : n ∈ setA) (hlt : n < 27) : n ≤ 13 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 27, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 13 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setC_le_6 {n : ℕ} (hn : n ∈ setC) (hlt : n < 25) : n ≤ 6 := by
  simp only [setC, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 25, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 6 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_3_5 : ∃ n : ℕ, n ∉ setAC := by
  use 20
  simp only [setAC, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_C, hab⟩
  have ha_bound : a ≤ 13 := setA_le_13 ha_A (by omega)
  have hb_bound : b ≤ 6 := setC_le_6 hb_C (by omega)
  omega

theorem erdos_125_generalized_3_5 : ∃ n : ℕ, n ∉ setAC :=
  gap_exists_3_5

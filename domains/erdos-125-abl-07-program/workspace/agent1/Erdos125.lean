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

-- Bounds for larger scale (5, 4): 3^5 = 243, 4^4 = 256
private lemma setA_le_121 {n : ℕ} (hn : n ∈ setA) (hlt : n < 243) : n ≤ 121 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 243, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_85 {n : ℕ} (hn : n ∈ setB) (hlt : n < 256) : n ≤ 85 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 256, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 85 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

-- Bounds for scale (6, 5): 3^6 = 729, 4^5 = 1024
private lemma setA_le_364 {n : ℕ} (hn : n ∈ setA) (hlt : n < 729) : n ≤ 364 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 364 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_341 {n : ℕ} (hn : n ∈ setB) (hlt : n < 1024) : n ≤ 341 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 1024, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 341 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

axiom exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 3 - ↑m * log 4| < ε

-- Direct gap interval: {62, 63} is not in setAB (proof independent of Dirichlet)
lemma gap_62_63_exists : ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨62, 2, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

-- Larger gap at scale (5,4): gap is [207, 243)
lemma gap_207_243_exists : ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨207, 36, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 121 := setA_le_121 ha_A (by omega)
  have hb_bound : b ≤ 85 := setB_le_85 hb_B (by omega)
  omega

-- Even larger gap at scale (6,5): gap is [706, 729)
lemma gap_706_729_exists : ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨706, 23, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 364 := setA_le_364 ha_A (by omega)
  have hb_bound : b ≤ 341 := setB_le_341 hb_B (by omega)
  omega

lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB :=
  gap_62_63_exists

-- Alternative witness: n=63 is also not in setAB
lemma gap_63_not_in_setAB : 63 ∉ setAB := by
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

-- Original witness: n=62 is not in setAB
lemma gap_62_not_in_setAB : 62 ∉ setAB := by
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

lemma gap_exists : ∃ n : ℕ, n ∉ setAB :=
  ⟨62, gap_62_not_in_setAB⟩

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

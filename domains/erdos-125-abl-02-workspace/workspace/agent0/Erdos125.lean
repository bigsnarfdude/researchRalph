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

-- PHASE 2: Generalization to bases (3, 5)
-- Both bases give restricted sets, unlike (2,3)

def setA35 : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB35 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setAB35 : Set ℕ := {n | ∃ a ∈ setA35, ∃ b ∈ setB35, a + b = n}

private lemma setA35_le_40 {n : ℕ} (hn : n ∈ setA35) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA35, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB35_le_31 {n : ℕ} (hn : n ∈ setB35) (hlt : n < 125) : n ≤ 31 := by
  simp only [setB35, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_35 : ∃ n : ℕ, n ∉ setAB35 := by
  use 72
  simp only [setAB35, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA35_le_40 ha_A (by omega)
  have hb_bound : b ≤ 31 := setB35_le_31 hb_B (by omega)
  omega

-- PHASE 2 continued: Generalization to bases (3, 7)
-- B's range must widen to 3 digits (7^3=343, max 57) but that pushes the gap
-- past 81, so A also needs a wider bound (3^5=243, max 121) to cover it.

def setB37 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setAB37 : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB37, a + b = n}

private lemma setA_le_121 {n : ℕ} (hn : n ∈ setA) (hlt : n < 243) : n ≤ 121 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 243, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB37_le_57 {n : ℕ} (hn : n ∈ setB37) (hlt : n < 343) : n ≤ 57 := by
  simp only [setB37, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_37 : ∃ n : ℕ, n ∉ setAB37 := by
  use 179
  simp only [setAB37, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 121 := setA_le_121 ha_A (by omega)
  have hb_bound : b ≤ 57 := setB37_le_57 hb_B (by omega)
  omega

-- PHASE 2 continued: Generalization to bases (5, 7) — neither base is 3,
-- confirms the technique doesn't depend on A being fixed at base 3.

def setA57 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setB57 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setAB57 : Set ℕ := {n | ∃ a ∈ setA57, ∃ b ∈ setB57, a + b = n}

private lemma setA57_le_31 {n : ℕ} (hn : n ∈ setA57) (hlt : n < 125) : n ≤ 31 := by
  simp only [setA57, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB57_le_57 {n : ℕ} (hn : n ∈ setB57) (hlt : n < 343) : n ≤ 57 := by
  simp only [setB57, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_57 : ∃ n : ℕ, n ∉ setAB57 := by
  use 89
  simp only [setAB57, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 31 := setA57_le_31 ha_A (by omega)
  have hb_bound : b ≤ 57 := setB57_le_57 hb_B (by omega)
  omega

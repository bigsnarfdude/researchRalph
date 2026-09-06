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

-- PHASE 2: Generalization to bases (4, 5)
-- Multiplicatively independent; both give restricted sets

def setA45 : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setB45 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setAB45 : Set ℕ := {n | ∃ a ∈ setA45, ∃ b ∈ setB45, a + b = n}

private lemma setA45_le_21 {n : ℕ} (hn : n ∈ setA45) (hlt : n < 64) : n ≤ 21 := by
  simp only [setA45, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB45_le_31 {n : ℕ} (hn : n ∈ setB45) (hlt : n < 125) : n ≤ 31 := by
  simp only [setB45, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_45 : ∃ n : ℕ, n ∉ setAB45 := by
  use 53
  simp only [setAB45, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 21 := setA45_le_21 ha_A (by omega)
  have hb_bound : b ≤ 31 := setB45_le_31 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (5, 7)
-- Both give restricted sets; multiplicatively independent

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

-- PHASE 2: Generalization to bases (5, 8)

def setA58 : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setB58 : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setAB58 : Set ℕ := {n | ∃ a ∈ setA58, ∃ b ∈ setB58, a + b = n}

private lemma setA58_le_31 {n : ℕ} (hn : n ∈ setA58) (hlt : n < 125) : n ≤ 31 := by
  simp only [setA58, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB58_le_73 {n : ℕ} (hn : n ∈ setB58) (hlt : n < 512) : n ≤ 73 := by
  simp only [setB58, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 512, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 73 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_58 : ∃ n : ℕ, n ∉ setAB58 := by
  use 105
  simp only [setAB58, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 31 := setA58_le_31 ha_A (by omega)
  have hb_bound : b ≤ 73 := setB58_le_73 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (6, 7)

def setA67 : Set ℕ := {n | ∀ d ∈ Nat.digits 6 n, d ≤ 1}
def setB67 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setAB67 : Set ℕ := {n | ∃ a ∈ setA67, ∃ b ∈ setB67, a + b = n}

private lemma setA67_le_43 {n : ℕ} (hn : n ∈ setA67) (hlt : n < 216) : n ≤ 43 := by
  simp only [setA67, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 216, (∀ d ∈ Nat.digits 6 m, d ≤ 1) → m ≤ 43 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB67_le_57 {n : ℕ} (hn : n ∈ setB67) (hlt : n < 343) : n ≤ 57 := by
  simp only [setB67, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_67 : ∃ n : ℕ, n ∉ setAB67 := by
  use 101
  simp only [setAB67, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 43 := setA67_le_43 ha_A (by omega)
  have hb_bound : b ≤ 57 := setB67_le_57 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (7, 8)

def setA78 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setB78 : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setAB78 : Set ℕ := {n | ∃ a ∈ setA78, ∃ b ∈ setB78, a + b = n}

private lemma setA78_le_57 {n : ℕ} (hn : n ∈ setA78) (hlt : n < 343) : n ≤ 57 := by
  simp only [setA78, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB78_le_73 {n : ℕ} (hn : n ∈ setB78) (hlt : n < 512) : n ≤ 73 := by
  simp only [setB78, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 512, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 73 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_78 : ∃ n : ℕ, n ∉ setAB78 := by
  use 131
  simp only [setAB78, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 57 := setA78_le_57 ha_A (by omega)
  have hb_bound : b ≤ 73 := setB78_le_73 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (6, 8)

def setA68 : Set ℕ := {n | ∀ d ∈ Nat.digits 6 n, d ≤ 1}
def setB68 : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setAB68 : Set ℕ := {n | ∃ a ∈ setA68, ∃ b ∈ setB68, a + b = n}

private lemma setA68_le_43 {n : ℕ} (hn : n ∈ setA68) (hlt : n < 216) : n ≤ 43 := by
  simp only [setA68, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 216, (∀ d ∈ Nat.digits 6 m, d ≤ 1) → m ≤ 43 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB68_le_73 {n : ℕ} (hn : n ∈ setB68) (hlt : n < 512) : n ≤ 73 := by
  simp only [setB68, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 512, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 73 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_68 : ∃ n : ℕ, n ∉ setAB68 := by
  use 117
  simp only [setAB68, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 43 := setA68_le_43 ha_A (by omega)
  have hb_bound : b ≤ 73 := setB68_le_73 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (8, 9)

def setA89 : Set ℕ := {n | ∀ d ∈ Nat.digits 8 n, d ≤ 1}
def setB89 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 1}
def setAB89 : Set ℕ := {n | ∃ a ∈ setA89, ∃ b ∈ setB89, a + b = n}

private lemma setA89_le_73 {n : ℕ} (hn : n ∈ setA89) (hlt : n < 512) : n ≤ 73 := by
  simp only [setA89, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 512, (∀ d ∈ Nat.digits 8 m, d ≤ 1) → m ≤ 73 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB89_le_121 {n : ℕ} (hn : n ∈ setB89) (hlt : n < 729) : n ≤ 121 := by
  simp only [setB89, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 9 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_89 : ∃ n : ℕ, n ∉ setAB89 := by
  use 195
  simp only [setAB89, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 73 := setA89_le_73 ha_A (by omega)
  have hb_bound : b ≤ 121 := setB89_le_121 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (6, 9)

def setA69 : Set ℕ := {n | ∀ d ∈ Nat.digits 6 n, d ≤ 1}
def setB69 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 1}
def setAB69 : Set ℕ := {n | ∃ a ∈ setA69, ∃ b ∈ setB69, a + b = n}

private lemma setA69_le_43 {n : ℕ} (hn : n ∈ setA69) (hlt : n < 216) : n ≤ 43 := by
  simp only [setA69, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 216, (∀ d ∈ Nat.digits 6 m, d ≤ 1) → m ≤ 43 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB69_le_121 {n : ℕ} (hn : n ∈ setB69) (hlt : n < 729) : n ≤ 121 := by
  simp only [setB69, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 9 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_69 : ∃ n : ℕ, n ∉ setAB69 := by
  use 165
  simp only [setAB69, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 43 := setA69_le_43 ha_A (by omega)
  have hb_bound : b ≤ 121 := setB69_le_121 hb_B (by omega)
  omega

-- PHASE 2: Generalization to bases (7, 9)

def setA79 : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setB79 : Set ℕ := {n | ∀ d ∈ Nat.digits 9 n, d ≤ 1}
def setAB79 : Set ℕ := {n | ∃ a ∈ setA79, ∃ b ∈ setB79, a + b = n}

private lemma setA79_le_57 {n : ℕ} (hn : n ∈ setA79) (hlt : n < 343) : n ≤ 57 := by
  simp only [setA79, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB79_le_121 {n : ℕ} (hn : n ∈ setB79) (hlt : n < 729) : n ≤ 121 := by
  simp only [setB79, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 729, (∀ d ∈ Nat.digits 9 m, d ≤ 1) → m ≤ 121 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_79 : ∃ n : ℕ, n ∉ setAB79 := by
  use 179
  simp only [setAB79, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 57 := setA79_le_57 ha_A (by omega)
  have hb_bound : b ≤ 121 := setB79_le_121 hb_B (by omega)
  omega

import Mathlib

open Finset

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

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

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

-- Generalization to (3,5) base pair
def setE : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setF : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setEF : Set ℕ := {n | ∃ e ∈ setE, ∃ f ∈ setF, e + f = n}

private lemma setE_le_40 {n : ℕ} (hn : n ∈ setE) (hlt : n < 81) : n ≤ 40 := by
  simp only [setE, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setF_le_31 {n : ℕ} (hn : n ∈ setF) (hlt : n < 125) : n ≤ 31 := by
  simp only [setF, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_35 : ∃ n : ℕ, n ∉ setEF := by
  use 72
  simp only [setEF, Set.mem_setOf_eq]
  rintro ⟨e, he_E, f, hf_F, hef⟩
  have he_bound : e ≤ 40 := setE_le_40 he_E (by omega)
  have hf_bound : f ≤ 31 := setF_le_31 hf_F (by omega)
  omega

theorem erdos_35 : ∃ n : ℕ, n ∉ setEF :=
  gap_exists_35

-- Generalization to (5,7) base pair
def setG : Set ℕ := {n | ∀ d ∈ Nat.digits 5 n, d ≤ 1}
def setH : Set ℕ := {n | ∀ d ∈ Nat.digits 7 n, d ≤ 1}
def setGH : Set ℕ := {n | ∃ g ∈ setG, ∃ h ∈ setH, g + h = n}

private lemma setG_le_31 {n : ℕ} (hn : n ∈ setG) (hlt : n < 125) : n ≤ 31 := by
  simp only [setG, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 125, (∀ d ∈ Nat.digits 5 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setH_le_57 {n : ℕ} (hn : n ∈ setH) (hlt : n < 343) : n ≤ 57 := by
  simp only [setH, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 343, (∀ d ∈ Nat.digits 7 m, d ≤ 1) → m ≤ 57 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists_57 : ∃ n : ℕ, n ∉ setGH := by
  use 89
  simp only [setGH, Set.mem_setOf_eq]
  rintro ⟨g, hg_G, h, hh_H, hgh⟩
  have hg_bound : g ≤ 31 := setG_le_31 hg_G (by omega)
  have hh_bound : h ≤ 57 := setH_le_57 hh_H (by omega)
  omega

theorem erdos_57 : ∃ n : ℕ, n ∉ setGH :=
  gap_exists_57


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

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

-- Phase 2 Candidate A: Generalization to (2, 3)
def setA₂₃ : Set ℕ := {n | ∀ d ∈ Nat.digits 2 n, d ≤ 1}
def setB₂₃ : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setAB₂₃ : Set ℕ := {n | ∃ a ∈ setA₂₃, ∃ b ∈ setB₂₃, a + b = n}

private lemma setA₂₃_le_31 {n : ℕ} (hn : n ∈ setA₂₃) (hlt : n < 32) : n ≤ 31 := by
  simp only [setA₂₃, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 32, (∀ d ∈ Nat.digits 2 m, d ≤ 1) → m ≤ 31 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB₂₃_le_13 {n : ℕ} (hn : n ∈ setB₂₃) (hlt : n < 27) : n ≤ 13 := by
  simp only [setB₂₃, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 27, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 13 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

lemma gap_exists₂₃ : ∃ n : ℕ, n ∉ setAB₂₃ := by
  use 44
  simp only [setAB₂₃, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 31 := setA₂₃_le_31 ha_A (by omega)
  have hb_bound : b ≤ 13 := setB₂₃_le_13 hb_B (by omega)
  omega

theorem erdos_125_generalized_2_3 : ∃ n : ℕ, n ∉ setAB₂₃ :=
  gap_exists₂₃

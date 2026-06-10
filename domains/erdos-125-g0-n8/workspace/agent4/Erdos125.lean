import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Key bounds derived computationally
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

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩
  -- a ∈ setA and a < 81 implies a ≤ 40
  have ha_le : a ≤ 40 := setA_le_40 ha (by omega)
  -- b ∈ setB and b < 64 implies b ≤ 21
  -- But b = 62 - a ≥ 62 - 40 = 22, contradiction
  have : b = 62 - a := by omega
  have hb_ge : b ≥ 22 := by omega
  have : b < 64 := by omega
  have hb_le : b ≤ 21 := setB_le_21 hb this
  omega

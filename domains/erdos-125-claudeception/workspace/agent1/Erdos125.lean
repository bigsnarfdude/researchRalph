import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Define the bounded finite sets
def elementsA : Finset ℕ := Finset.filter (fun n => ∀ d ∈ Nat.digits 3 n, d ≤ 1) (Finset.range 81)
def elementsB : Finset ℕ := Finset.filter (fun n => ∀ d ∈ Nat.digits 4 n, d ≤ 1) (Finset.range 64)

-- Aliases for compatibility
def setA_bounded := elementsA
def setB_bounded := elementsB

-- Helper lemma: if a number has digit 2 in base-3, it's not in setA
lemma has_digit_2_base3_not_in_setA {n : ℕ} (h : 2 ∈ Nat.digits 3 n) : n ∉ setA := by
  intro hn
  simp [setA, Set.mem_setOf] at hn
  exact absurd (hn 2 h) (by norm_num)

lemma setA_in_bounded : setA ⊆ setOf (· ∈ setA_bounded) := by
  intro n hn
  simp [setA_bounded, Finset.mem_filter, Finset.mem_range, setA] at *
  constructor
  · omega  -- setA elements up to 62 are < 81
  · exact hn

lemma setB_in_bounded : setB ⊆ setOf (· ∈ setB_bounded) := by
  intro n hn
  simp [setB_bounded, Finset.mem_filter, Finset.mem_range, setB] at *
  constructor
  · omega  -- setB elements up to 62 are < 64
  · exact hn

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro h
  simp only [setAB, Set.mem_setOf] at h
  obtain ⟨a, ha, b, hb, hab⟩ := h

  have b_bound : b ≤ 62 := by omega

  -- Case analysis on b: for each b ≤ 62, derive False
  interval_cases b
  all_goals (
    -- Expand the digit constraints
    simp only [setA, setB, Set.mem_setOf] at ha hb
    -- ha : ∀ d ∈ Nat.digits 3 a, d ≤ 1
    -- hb : ∀ d ∈ Nat.digits 4 b, d ≤ 1
    -- hab : a + b = 62

    -- Try to show one of the constraints is false
    (norm_num [Nat.digits] at hb) <|>
    (have eq_a : a = 62 - b := by omega
     rw [eq_a] at ha
     norm_num [Nat.digits] at ha)
  )

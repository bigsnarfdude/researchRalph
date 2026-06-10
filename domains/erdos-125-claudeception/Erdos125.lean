import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf, not_exists]
  intro a ⟨ha, b, hb, hab⟩

  have a_le : a ≤ 62 := by omega
  have b_eq : b = 62 - a := by omega

  -- Try interval_cases followed by explicit refutation
  interval_cases a
  all_goals (
    -- First try to refute ha (claim that a is in setA)
    (try (simp only [setA, Set.mem_setOf, Nat.digits] at ha; norm_num [Nat.digitsAux] at ha))
    -- If that doesn't work, try to refute hb (claim that b is in setB)
    (try (simp only [setB, Set.mem_setOf, Nat.digits, b_eq] at hb; norm_num [Nat.digitsAux] at hb))
  )

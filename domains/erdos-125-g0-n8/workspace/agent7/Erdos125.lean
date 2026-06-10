import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf, not_exists, not_and]
  intro a ha b hb hab
  have hb_bound : b ≤ 62 := by omega
  interval_cases b <;> (
    simp only [setA, setB, Set.mem_setOf, Nat.digits] at ha hb
    norm_num at hb ha
  )

import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩

  -- Extract bounds from equation and membership
  have a_lt : a < 81 := by omega
  have b_lt : b < 64 := by omega

  -- Now interval_cases over all possibilities
  interval_cases a
  -- This should create 81 sub-goals, one for each a from 0 to 80
  -- For each a, check all b from 0 to 63
  all_goals (
    interval_cases b <|> (
      -- For each (a,b) pair, check if both are in their sets and a+b=62
      -- If any is not in its set, we have a contradiction with ha or hb
      -- If a+b≠62, we have contradiction with hab
      simp only [setA, setB, Set.mem_setOf] at ha hb
      omega
    )
  )

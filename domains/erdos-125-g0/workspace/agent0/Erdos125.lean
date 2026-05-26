import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  intro ⟨a, ha, b, hb, hab⟩

  -- We have a ∈ setA, b ∈ setB, and a + b = 62
  -- The only elements of setA ≤ 62 are: {0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40}
  -- The only elements of setB ≤ 62 are: {0, 1, 4, 5, 16, 17, 20, 21}

  -- Since a ≤ 62 and a + b = 62, we have b ≥ 0
  -- Also since a ∈ setA, a must be one of the above 16 values
  -- For each value of a, we need b = 62 - a to not be in setB

  -- This is decidable: we enumerate cases and check each one
  have ha_bound : a ≤ 40 := by
    -- If a ∈ setA and a > 40, then a ≥ 45 (next element of setA)
    -- But then b = 62 - a ≤ 17, and we'd need b ∈ setB
    -- However, we can verify no element of setA > 40 and ≤ 62 forms a valid pair
    omega

  -- Now case-split on a ∈ {0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40}
  -- For each case, show that 62 - a ∉ setB

  interval_cases a <;> (try simp [setA, setB, Nat.digits] at ha hb hab) <;> (try omega)

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

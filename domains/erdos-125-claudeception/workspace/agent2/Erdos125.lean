import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Decidable versions for computation
def inSetA (n : ℕ) : Bool := (Nat.digits 3 n).all (· ≤ 1)
def inSetB (n : ℕ) : Bool := (Nat.digits 4 n).all (· ≤ 1)

lemma inSetA_iff (n : ℕ) : inSetA n ↔ n ∈ setA := by
  simp [inSetA, setA, List.all_eq_true]

lemma inSetB_iff (n : ℕ) : inSetB n ↔ n ∈ setB := by
  simp [inSetB, setB, List.all_eq_true]

-- Compute the elements of setA and setB up to a bound
def setA_list (bound : ℕ) : List ℕ :=
  (List.range (bound + 1)).filter inSetA

def setB_list (bound : ℕ) : List ℕ :=
  (List.range (bound + 1)).filter inSetB

-- Compute all possible sums
def sums_list (bound : ℕ) : List ℕ :=
  let a_vals := setA_list bound
  let b_vals := setB_list bound
  List.deduplicate (a_vals.flatMap fun a => b_vals.map (a + ·))

-- Verify that 62 is not a sum
lemma check_62_not_in_sums : ¬(62 ∈ sums_list 65) := by
  native_decide

-- Connect back to setAB
lemma not_62_in_setAB : 62 ∉ setAB := by
  intro ⟨a, ha, b, hb, hab⟩
  -- a + b = 62, a ∈ setA, b ∈ setB
  -- Since a + b = 62 and both are ≤ 62, we have a, b ≤ 62
  have ha_in_list : a ∈ setA_list 62 := by
    simp [setA_list, inSetA_iff]
    exact ⟨by omega, ha⟩
  have hb_in_list : b ∈ setB_list 62 := by
    simp [setB_list, inSetB_iff]
    exact ⟨by omega, hb⟩
  -- Then a + b should be in the sums_list
  have : 62 ∈ sums_list 62 := by
    sorry  -- This requires showing that all sums are in the list
  exact check_62_not_in_sums this

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  exact not_62_in_setAB

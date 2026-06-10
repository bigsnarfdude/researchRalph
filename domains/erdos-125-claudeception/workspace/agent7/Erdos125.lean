import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

-- Decidable version of setB membership for bounds
def inSetB_dec (n : ℕ) : Bool :=
  (Nat.digits 4 n).all fun d => d ≤ 1

-- Key insight: if 62 - b has a base-3 digit of 2, then 62 - b ∉ setA
lemma not_in_setA_if_has_digit_2 (n : ℕ) (h : 2 ∈ Nat.digits 3 n) : n ∉ setA := by
  unfold setA
  simp only [Set.mem_setOf]
  push_neg
  exact ⟨2, h, by norm_num⟩

-- Computational verification that digit 2 appears in each case
example : (2 : ℕ) ∈ Nat.digits 3 62 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 61 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 58 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 57 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 46 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 45 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 42 := by decide
example : (2 : ℕ) ∈ Nat.digits 3 41 := by decide

-- Handle each value of b: must verify computationally
lemma b_case_0 (h : 0 ∈ setB) : (62 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_1 (h : 1 ∈ setB) : (61 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

-- More cases for completeness
lemma b_case_4 (h : 4 ∈ setB) : (58 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_5 (h : 5 ∈ setB) : (57 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_16 (h : 16 ∈ setB) : (46 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_17 (h : 17 ∈ setB) : (45 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_20 (h : 20 ∈ setB) : (42 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

lemma b_case_21 (h : 21 ∈ setB) : (41 : ℕ) ∉ setA := by
  apply not_in_setA_if_has_digit_2
  decide

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  unfold setAB setA setB
  simp only [Set.mem_setOf]
  push_neg
  intro a ha b hb
  -- Match on possible values of b (limited by digit constraint hb)
  by_contra h_eq
  push_neg at h_eq
  -- Now we have a + b = 62 and need to derive contradiction
  -- Using the fact that b ∈ setB severely limits b's value
  omega

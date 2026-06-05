import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- YOUR TASK: implement the construction described in program.md and prove the theorem below.
-- Read mathlib_hints.md before you start — it lists the exact Mathlib lemmas you need.

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, {ck k} ∪ Bk k ∪ Fk k

-- Partial union up through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  simp only [Q]
  norm_num [pow_succ]

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ, Nat.mul_comm]

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  simp only [Akn, Set.mem_union, hx, true_or]

-- Helper: 2 + 3 = 5, so we can represent 4 as 2 + 2
lemma base_case_4 : ∃ a ∈ Akn 1, ∃ b ∈ Akn 1, a + b = 4 := by
  use 2, by simp [Akn]
  use 2, by simp [Akn]
  norm_num

lemma basis_lem (k : ℕ) : ∀ n : ℕ, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = n := by
  sorry

lemma rigidity (k : ℕ) : ∀ n : ℕ, n ∈ Jk k → ∀ a b : ℕ, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

-- Helper lemma: Akn k is a subset of setA
lemma Akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  intro x hx
  induction k with
  | zero =>
    simp only [Akn] at hx
    simp [setA, hx]
  | succ k ih =>
    simp only [Akn] at hx
    simp only [Set.mem_union] at hx
    rcases hx with h | h | h | h
    · exact ih h
    · simp [setA, Set.mem_iUnion]
      use k
      simp [h]
    · simp [setA, Set.mem_iUnion]
      use k
      simp [Set.mem_union, h]
    · simp [setA, Set.mem_iUnion]
      use k
      simp [Set.mem_union, h]

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  use setA
  constructor
  · -- Basis property
    intro n hn
    obtain ⟨a, ha, b, hb, hab⟩ := basis_lem 0 n hn (by omega : n ≤ 6 * Q 0)
    exact ⟨a, Akn_subset_setA 1 ha, b, Akn_subset_setA 1 hb, hab⟩
  · -- No partition can have both parts syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨C₁, hC₁⟩ ⟨C₂, hC₂⟩
    -- For large enough k, ck k ∈ A and Jk k ∩ (A₂ + A₂) is empty
    sorry

end Erdos741OAI

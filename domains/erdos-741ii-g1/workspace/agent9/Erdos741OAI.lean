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

def Q (k : ℕ) := 5 ^ k

def ck (k : ℕ) := 4 * Q k

def Bk (k : ℕ) := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

lemma akn_mono (j k : ℕ) (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction k generalizing j with
  | zero =>
    have : j = 0 := by omega
    rw [this]
  | succ k ih =>
    by_cases hj : j ≤ k
    · have h1 := ih hj
      have h2 : Akn k ⊆ Akn (k + 1) := by
        unfold Akn
        intro x hx
        right
        exact Or.inl hx
      exact Subset.trans h1 h2
    · push_neg at hj
      have : j = k + 1 := by omega
      rw [this]

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

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
  · intro n hn
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    sorry

end Erdos741OAI

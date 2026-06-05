import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k

def Bk : ℕ → Set ℕ := fun k => {n : ℕ | 5 * Q k ≤ n ∧ n ≤ 6 * Q k - 1}

def Fk : ℕ → Set ℕ := fun k => {n : ℕ | 10 * Q k - 1 ≤ n ∧ n ≤ 15 * Q k}

def Jk : ℕ → Set ℕ := fun k => {n : ℕ | 9 * Q k ≤ n ∧ n < 10 * Q k}

def stagek : ℕ → Set ℕ := fun k => {n : ℕ | n = ck k ∨ n ∈ Bk k ∨ n ∈ Fk k}

def setA : Set ℕ := {x : ℕ | x = 2 ∨ x = 3 ∨ ∃ k, x ∈ stagek k}

-- Partial union for induction
def Akn : ℕ → Set ℕ
  | 0 => {x : ℕ | x = 2 ∨ x = 3}
  | k + 1 => Akn k ∪ stagek k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_one : Q 1 = 5 := by norm_num [Q, pow_succ]

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  ring

-- Akn is monotone and unions to form setA
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  simp only [Akn]
  exact fun x hx => Or.inl hx

-- Every element of Akn is in setA
lemma akn_in_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry

-- Basis lemma (main content)
-- Every n ≥ 4 is a sum of two elements from setA
-- Proof: By strong induction, show n ∈ Icc 4 (6 * Q k) for some k,
-- then cover by 8 pair types (I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk)
lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
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
  · exact basis_lem
  · intro A₁ A₂ _hA₁ _hA₂ hpart hdisj h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2
    sorry

end Erdos741OAI

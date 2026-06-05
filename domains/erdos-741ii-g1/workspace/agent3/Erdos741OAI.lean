import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

-- Main basis lemma: Icc 4 (6*Q k) ⊆ Akn (k+1) + Akn (k+1)
lemma basis_lem (k : ℕ) : ∀ x, x ∈ Icc 4 (6 * Q k) → x ∈ Akn (k + 1) + Akn (k + 1) := by
  sorry

-- Rigidity lemma: elements in Jk that sum in A come only from ck + Bk
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (ha : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n) :
    ∃ (a b : ℕ), ((a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)) ∧ a + b = n := by
  sorry

-- Gap lemma: if ck is not in T, then no element of Jk can be a sum from T
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
  · -- setA is a basis: all n ≥ 4 can be written as sum of two elements
    intro n hn
    sorry
  · -- No partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj hsynd
    -- Extract the gap bounds from the syndetic hypothesis
    unfold IsSyndetic at hsynd
    obtain ⟨C₁, hC₁⟩ := hsynd.1
    obtain ⟨C₂, hC₂⟩ := hsynd.2
    -- Find a k large enough that Q k > max(C₁, C₂)
    let k := max C₁ C₂ + 1
    -- ck k must be in A, so it's in one partition
    have hck_mem : ck k ∈ setA := by
      unfold setA
      right
      refine ⟨k, ?_⟩
      left
      rfl
    have hck_in : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_mem
    -- If ck k ∈ A₁, then A₂ + A₂ avoids Jk k
    -- But C₂-syndetic means A₂ + A₂ hits [9*Q k, 9*Q k + C₂]
    -- which is a subset of Jk k — contradiction
    sorry

end Erdos741OAI

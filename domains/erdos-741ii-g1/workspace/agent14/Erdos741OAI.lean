import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k

def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)

def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)

def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def Lk : ℕ → Set ℕ := fun k => Icc (2 * Q k) (3 * Q k)

def stageK : ℕ → Set ℕ := fun k => {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stageK k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ stageK k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ]
  ring

lemma Q_mono : ∀ j k, j < k → Q j < Q k := by
  intros j k hjk
  unfold Q
  exact Nat.pow_lt_pow_right (by norm_num : 1 < 5) hjk

lemma akn_mono : ∀ j k, j ≤ k → Akn j ⊆ Akn k := by
  intro j k hjk
  induction k generalizing j with
  | zero =>
    have : j = 0 := by omega
    rw [this]
  | succ k ih =>
    by_cases h : j ≤ k
    · have := ih j h
      simp only [Akn] at this ⊢
      exact fun x hx => Or.inl (this hx)
    · push_neg at h
      have : j = k + 1 := by omega
      rw [this]

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  sorry

lemma Lk_in_setA (k : ℕ) : Lk k ⊆ setA := by
  sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (ha : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n) :
    ∃ a b, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) ∧ a + b = n := by
  obtain ⟨a, ha_mem, b, hb_mem, hab⟩ := ha
  simp only [Jk, mem_Ico] at hn
  use a, b
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck_not_in : ck k ∉ T) :
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
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    sorry

end Erdos741OAI

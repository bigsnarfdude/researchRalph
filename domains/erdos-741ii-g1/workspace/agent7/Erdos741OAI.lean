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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_mono {k₁ k₂ : ℕ} (h : k₁ ≤ k₂) : Akn k₁ ⊆ Akn k₂ := by
  revert k₁
  induction k₂ with
  | zero =>
    intro k₁ h x hx
    simp [Nat.le_zero] at h
    rw [h]
  | succ k₂ ih =>
    intro k₁ h x hx
    omega

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc, Set.mem_add] at hx ⊢
  by_cases h1 : x ≤ 5 * Q k
  · use x - 2*Q k
    constructor
    · sorry
    · use 2*Q k
      constructor
      · sorry
      · sorry
  · by_cases h2 : x ≤ 6 * Q k
    · use x - 5*Q k
      constructor
      · sorry
      · use 5*Q k
        constructor
        · sorry
        · sorry
    · omega

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
    by_cases h : n < 4
    · omega
    · push_neg at h
      -- There exists k with n ≤ 6 * Q k
      sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨C₁, hC₁⟩ ⟨C₂, hC₂⟩
    sorry

end Erdos741OAI

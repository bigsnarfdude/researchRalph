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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos : ∀ k, 0 < Q k := fun k => by
  unfold Q
  norm_num

lemma Q_one : Q 1 = 5 := by
  unfold Q
  norm_num

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := fun k => by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_subset : ∀ k, Akn k ⊆ setA := by
  intro k
  induction k with
  | zero =>
    intro x hx
    unfold Akn at hx
    unfold setA
    left
    exact hx
  | succ k ih =>
    intro x hx
    unfold Akn at hx
    unfold setA
    cases hx with
    | inl h =>
      left
      exact ih h
    | inr h =>
      right
      exact ⟨k, h⟩

lemma basis_lem : ∀ k, Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro k x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  rw [Set.mem_add]
  unfold Akn
  simp only [Set.mem_union, Set.mem_singleton_iff]
  -- x ∈ [4, 6*Q(k+1)], must write as sum from Akn(k) ∪ {ck(k)} ∪ Bk(k) ∪ Fk(k)
  sorry

lemma rigidity_lem : ∀ k,
    ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA,
    a + b = n → (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k
  sorry

lemma gap_lem : ∀ k, ∀ T ⊆ setA,
    ck k ∉ T →
    Jk k ∩ (T + T) = ∅ := by
  intro k T hT hck
  sorry

lemma basis_covers : ∀ n : ℕ, 4 ≤ n → ∃ k, n ≤ 6 * Q (k + 1) := by
  intro n hn
  by_cases h : n ≤ 30
  · use 0
    unfold Q
    omega
  · push_neg at h
    -- For n > 30, keep doubling k until n ≤ 6*Q(k+1)
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
    obtain ⟨k, hk⟩ := basis_covers n hn
    have hmem : n ∈ Icc 4 (6 * Q (k + 1)) := ⟨hn, hk⟩
    have := basis_lem k hmem
    obtain ⟨a, ha, b, hb, hab⟩ := this
    refine ⟨a, akn_subset (k + 1) a ha, b, akn_subset (k + 1) b hb, hab⟩
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    sorry

end Erdos741OAI

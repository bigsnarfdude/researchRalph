import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
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
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  ring

-- Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  simp only [Akn, Set.mem_union, Set.mem_singleton_iff] at hx ⊢
  tauto

-- Basis lemma: every n in [4, 6*Q k] can be written as sum of two elements from Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  sorry

-- Rigidity: elements from Jk that sum must involve ck
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- n ∈ [9Q k, 10Q k) and a+b = n, both in setA
  -- Case analysis: a is either {2,3}, stage j<k, j=k, or j>k
  -- Only a = ck k with b ∈ Bk k works (or vice versa)
  have : 9 * Q k ≤ n ∧ n < 10 * Q k := by
    simp only [Jk, mem_Ico] at hn
    exact hn
  rcases this with ⟨hn_lo, hn_hi⟩
  rw [← hab] at hn_lo hn_hi
  -- Now 9*Q k ≤ a+b < 10*Q k
  -- The only way this works is ck k + something from Bk k
  -- This requires careful case analysis on where a,b come from in setA
  sorry

-- Gap lemma: if ck k not in T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) : Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn_jk, hn_sum⟩
  simp only [Set.mem_add] at hn_sum
  rcases hn_sum with ⟨a, ha_mem, b, hb_mem, hab⟩
  have ha_setA : a ∈ setA := hT ha_mem
  have hb_setA : b ∈ setA := hT hb_mem
  have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := rigidity_lem k n hn_jk a b ha_setA hb_setA hab
  rcases this with ⟨eq_a, _⟩ | ⟨eq_b, _⟩
  · rw [eq_a] at ha_mem
    exact hck ha_mem
  · rw [eq_b] at hb_mem
    exact hck hb_mem

-- Helper: setA contains 2 and 3
lemma two_in_setA : (2 : ℕ) ∈ setA := by
  simp [setA, Set.mem_union, Set.mem_singleton_iff]

lemma three_in_setA : (3 : ℕ) ∈ setA := by
  simp [setA, Set.mem_union, Set.mem_singleton_iff]

-- Helper: elements of Akn are in setA
lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
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
  · intro A₁ A₂ h1 h2 hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    sorry

end Erdos741OAI

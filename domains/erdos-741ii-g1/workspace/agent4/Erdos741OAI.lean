import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- CONSTRUCTION
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- HELPER LEMMAS
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  calc Q (k + 1) = 5 ^ (k + 1) := rfl
    _ = 5 * 5 ^ k := by ring
    _ = 5 * Q k := rfl

lemma akn_mono {k₁ k₂ : ℕ} (h : k₁ ≤ k₂) : Akn k₁ ⊆ Akn k₂ := by sorry

-- Akn is a subset of setA
lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by sorry

-- BASIS LEMMA: every n ∈ [4, 6*Q k] is a sum of two elements from Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro n hn
    simp only [mem_Icc, Q, Set.mem_add] at *
    have hn_lo : 4 ≤ n := hn.1
    have hn_hi : n ≤ 6 := by omega
    interval_cases n
    · exact ⟨2, by left; norm_num [Akn], 2, by left; norm_num [Akn], by norm_num⟩
    · exact ⟨2, by left; norm_num [Akn], 3, by left; norm_num [Akn], by norm_num⟩
    · exact ⟨3, by left; norm_num [Akn], 3, by left; norm_num [Akn], by norm_num⟩
  | succ k ih =>
    intro n hn
    -- Apply the inductive hypothesis for Akn k to get some elements that sum to values in [4, 6*Q k]
    -- Then use these to build sums for values in [4, 6*Q(k+1)]
    -- Case analysis on which interval n falls into
    sorry

-- RIGIDITY LEMMA: for n ∈ Jk(k), the only way to write n as a sum from A is with ck(k)
lemma rigidity_lem (k : ℕ) : ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Helper: large enough k gives us the bounds we need
lemma Q_large_enough (C k : ℕ) (h : C < Q k) : 9 * Q k + C < 10 * Q k := by
  omega

-- GAP LEMMA: if ck(k) ∉ T, then Jk(k) ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck : ck k ∉ T) :
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
  · -- Basis: every n ≥ 4 is a sum of two elements from setA
    intro n hn
    -- Find k such that n ∈ [4, 6*Q k]
    have : ∃ k, n ∈ Icc 4 (6 * Q k) := by sorry
    rcases this with ⟨k, hk⟩
    -- Use basis_lem to find elements from Akn(k+1)
    have : n ∈ Akn (k + 1) + Akn (k + 1) := basis_lem k hk
    -- Akn(k+1) ⊆ setA
    have hAkn_sub : Akn (k + 1) ⊆ setA := by sorry
    -- Therefore n ∈ setA + setA
    exact Set.add_subset_add hAkn_sub hAkn_sub this
  · -- Partition rigidity
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- One of A₁, A₂ doesn't contain ck(max(C₁, C₂)+1)
    let k := max C₁ C₂ + 1
    have hQk_bound : max C₁ C₂ < Q k := by sorry
    have hck_in : ck k ∈ setA := by sorry
    have hck_part : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_in
    rcases hck_part with hck_A₁ | hck_A₂
    · -- ck k ∈ A₁, so ck k ∉ A₂
      have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck_A₁, h⟩
        rw [hdisj] at this
        simp at this
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck_not_A₂
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      rcases this with ⟨m, hm_sum, hm_Icc⟩
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        obtain ⟨hlo, hhi⟩ := hm_Icc
        constructor
        · exact hlo
        · have : 9 * Q k + C₂ < 10 * Q k := Q_large_enough C₂ k (by omega : C₂ < Q k)
          omega
      have hmem : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hm_in_Jk hm_sum
      rw [hgap] at hmem
      simp at hmem
    · -- ck k ∈ A₂, so ck k ∉ A₁
      have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck_A₂⟩
        rw [hdisj] at this
        simp at this
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck_not_A₁
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      rcases this with ⟨m, hm_sum, hm_Icc⟩
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        obtain ⟨hlo, hhi⟩ := hm_Icc
        constructor
        · exact hlo
        · have : 9 * Q k + C₁ < 10 * Q k := Q_large_enough C₁ k (by omega : C₁ < Q k)
          omega
      have hmem : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter hm_in_Jk hm_sum
      rw [hgap] at hmem
      simp at hmem

end Erdos741OAI

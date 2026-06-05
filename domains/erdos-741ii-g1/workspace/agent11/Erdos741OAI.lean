import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; ring

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry

lemma basis_lem (k : ℕ) : ∀ n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = n := by
  sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) :
    ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
      (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hc : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro hmem
  obtain ⟨hx_jk, a, ha, b, hb, hab⟩ := hmem
  have rig := rigidity k x hx_jk a (hT ha) b (hT hb) hab
  rcases rig with (⟨ha_ck, _⟩ | ⟨hb_ck, _⟩)
  · rw [ha_ck] at ha; exact hc ha
  · rw [hb_ck] at hb; exact hc hb

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
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj hsynd
    obtain ⟨C₁, hsyn₁⟩ := hsynd.1
    obtain ⟨C₂, hsyn₂⟩ := hsynd.2
    have : ∃ k, max C₁ C₂ < Q k := by sorry
    obtain ⟨k, hk⟩ := this
    have hck_A : ck k ∈ setA := by sorry
    have hck_part : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_A
    rcases hck_part with hck₁ | hck₂
    · have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck₁, h⟩
        rw [hdisj] at this
        exact absurd this (Set.mem_empty (ck k))
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck_not_A₂
      obtain ⟨m, hm_sum, hm_interval⟩ := hsyn₂ (9 * Q k)
      have : Icc (9 * Q k) (9 * Q k + C₂) ⊆ Jk k := by
        intro x hx
        unfold Jk
        simp only [Set.mem_Ico, Set.mem_Icc] at hx ⊢
        constructor
        · exact hx.1
        · omega
      have hm_in_Jk : m ∈ Jk k := this hm_interval
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hm_in_Jk hm_sum
      rw [hgap] at this
      exact absurd this (Set.mem_empty m)
    · have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck₂⟩
        rw [hdisj] at this
        exact absurd this (Set.mem_empty (ck k))
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck_not_A₁
      obtain ⟨m, hm_sum, hm_interval⟩ := hsyn₁ (9 * Q k)
      have : Icc (9 * Q k) (9 * Q k + C₁) ⊆ Jk k := by
        intro x hx
        unfold Jk
        simp only [Set.mem_Ico, Set.mem_Icc] at hx ⊢
        constructor
        · exact hx.1
        · omega
      have hm_in_Jk : m ∈ Jk k := this hm_interval
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter hm_in_Jk hm_sum
      rw [hgap] at this
      exact absurd this (Set.mem_empty m)

end Erdos741OAI

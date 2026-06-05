import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Construction
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_mono : ∀ j k, j ≤ k → Q j ≤ Q k := by
  intro j k hjk
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := by
  intro k x hx
  simp only [Akn] at hx ⊢
  cases k with
  | zero =>
    left
    exact hx
  | succ k =>
    simp only [Akn] at hx ⊢
    left
    exact hx

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc, mem_add] at hx ⊢
  obtain ⟨hx_lo, hx_hi⟩ := hx
  sorry

lemma mem_setA_or_base (n : ℕ) : n = 2 ∨ n = 3 ∨ ∃ k : ℕ, n ∈ {ck k} ∪ Bk k ∪ Fk k := by
  -- This is not needed for the main proof, but provided for structure
  sorry

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  unfold setA
  right
  simp only [Set.mem_iUnion]
  use k
  simp [Set.mem_union]

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Stage decomposition: a and b come from levels 0, 1, ..., k
  -- For n ∈ [9*Qk, 10*Qk), only ck k + something(Bk k) reaches this range
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
    -- Basis property: every n ≥ 4 is a sum of two elements from setA
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj hsynd
    -- Rigidity argument: no partition can have both parts syndetic
    obtain ⟨C₁, hC₁⟩ := hsynd.1
    obtain ⟨C₂, hC₂⟩ := hsynd.2
    -- ck k must be in one of the parts; use a large k
    let C := max C₁ C₂
    -- Pick k large enough: Q k > C
    have hC_lt : C < Q (C + 1) := by
      -- For all C, C < 5^(C+1)
      have key : ∀ n : ℕ, n < 5 ^ (n + 1) := by
        intro n
        induction n with
        | zero => norm_num
        | succ n ih =>
          have h1 : n < 5 ^ (n + 1) := ih
          have h2 : 5 ^ (n + 1) < 5 ^ (n + 2) := by
            have : 5 ^ (n + 1) * 1 < 5 ^ (n + 1) * 5 := by omega
            calc 5 ^ (n + 1) = 5 ^ (n + 1) * 1 := by ring
            _ < 5 ^ (n + 1) * 5 := this
            _ = 5 ^ (n + 2) := by simp [Nat.pow_succ, mul_comm]
          omega
      exact key C
    -- ck (C+1) is in setA, hence in A₁ or A₂
    have hck_mem : ck (C + 1) ∈ setA := ck_mem_setA (C + 1)
    have hck_partition : ck (C + 1) ∈ A₁ ∨ ck (C + 1) ∈ A₂ := by
      exact hpart (ck (C + 1)) hck_mem
    -- Case split: if ck (C+1) ∈ A₁, then A₂ + A₂ is empty on Jk (C+1)
    rcases hck_partition with h_in_A1 | h_in_A2
    · -- ck (C+1) ∈ A₁
      -- Then by gap_lem, Jk (C+1) ∩ (A₂ + A₂) = ∅
      have hck_not_A2 : ck (C + 1) ∉ A₂ := by
        intro h
        have : ck (C + 1) ∈ A₁ ∩ A₂ := Set.mem_inter h_in_A1 h
        exact absurd this (by simp [hdisj])
      have hgap := gap_lem (C + 1) A₂ hA₂ hck_not_A2
      -- But A₂ + A₂ is syndetic with bound C₂
      -- So it hits [9*Qk, 9*Qk + C₂] ⊆ Jk (C+1)
      have h_exist : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q (C + 1)) (9 * Q (C + 1) + C₂) := by
        exact hC₂ (9 * Q (C + 1))
      obtain ⟨m, hm_mem, hm_interval⟩ := h_exist
      -- But m ∈ Jk (C+1) by subset, contradicting the gap
      have hm_Jk : m ∈ Jk (C + 1) := by
        simp only [mem_Icc] at hm_interval
        unfold Jk
        simp only [mem_Ico]
        obtain ⟨hlo, hhi⟩ := hm_interval
        constructor
        · exact hlo
        · have : C₂ ≤ C := le_max_right C₁ C₂
          have : C < Q (C + 1) := hC_lt
          omega
      have h_contra : m ∈ Jk (C + 1) ∩ (A₂ + A₂) := Set.mem_inter hm_Jk hm_mem
      rw [hgap] at h_contra
      cases h_contra
    · -- ck (C+1) ∈ A₂, symmetric case
      have hck_not_A1 : ck (C + 1) ∉ A₁ := by
        intro h
        have : ck (C + 1) ∈ A₁ ∩ A₂ := Set.mem_inter h h_in_A2
        exact absurd this (by simp [hdisj])
      have hgap := gap_lem (C + 1) A₁ hA₁ hck_not_A1
      have h_exist : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q (C + 1)) (9 * Q (C + 1) + C₁) := by
        exact hC₁ (9 * Q (C + 1))
      obtain ⟨m, hm_mem, hm_interval⟩ := h_exist
      have hm_Jk : m ∈ Jk (C + 1) := by
        simp only [mem_Icc] at hm_interval
        unfold Jk
        simp only [mem_Ico]
        obtain ⟨hlo, hhi⟩ := hm_interval
        constructor
        · exact hlo
        · have : C₁ ≤ C := le_max_left C₁ C₂
          have : C < Q (C + 1) := hC_lt
          omega
      have h_contra : m ∈ Jk (C + 1) ∩ (A₁ + A₁) := Set.mem_inter hm_Jk hm_mem
      rw [hgap] at h_contra
      cases h_contra

end Erdos741OAI

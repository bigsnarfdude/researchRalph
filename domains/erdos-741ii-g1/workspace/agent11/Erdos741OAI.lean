import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Basic construction
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- Partial union up through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Full set A
def setA : Set ℕ := ⋃ k, Akn k

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]

lemma Bk_nonempty (k : ℕ) : (Bk k).Nonempty := by
  use 5 * Q k
  simp only [Bk, mem_Icc]
  constructor
  · omega
  · omega

lemma Fk_nonempty (k : ℕ) : (Fk k).Nonempty := by
  use 10 * Q k - 1
  simp only [Fk, mem_Icc]
  constructor
  · omega
  · omega

-- Monotonicity of Akn
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Akn] at hx ⊢
    left; exact hx
  | succ k ih =>
    intro x hx
    simp only [Akn] at hx ⊢
    have : x ∈ Akn (k + 1) := by
      induction (k + 1) with
      | zero => exact hx
      | succ j hij =>
        simp only [Akn] at hij ⊢
        left; exact hij
    left; exact this

-- The basis property
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  -- Use by_cases to split on intervals
  by_cases h1 : x ≤ 5 * Q k
  · -- x ∈ [4, 5*Qk], use (x-2*Qk) + 2*Qk where both in Akn(k+1)
    have h2 : 2 * Q k ≤ x := by omega
    use x - 2 * Q k
    constructor
    · -- x - 2*Qk ∈ Akn(k+1)
      apply akn_mono
      simp only [Akn]
      right; right; left
      simp only [ck, mem_singleton_iff]
      omega
    use 2 * Q k
    constructor
    · -- 2*Qk ∈ Akn(k+1)
      apply akn_mono
      simp only [Akn]
      left
      simp only [mem_insert_iff, mem_singleton_iff]
      right
    · omega
  · push_neg at h1
    -- x ∈ (5*Qk, 6*Qk]
    use x - 5 * Q k
    constructor
    · -- x - 5*Qk ∈ Akn(k+1)
      apply akn_mono
      simp only [Akn]
      left
      simp only [mem_insert_iff, mem_singleton_iff]
      left
    use 5 * Q k
    constructor
    · -- 5*Qk ∈ Akn(k+1)
      simp only [Akn]
      right; right; right; left
      simp only [Bk, mem_Icc]
      omega
    · omega

-- Rigidity: decomposition of sums in Jk k
lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
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
  · -- Basis property: every n ≥ 4 is a sum
    intro n hn
    sorry
  · -- No partition is both syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    sorry

end Erdos741OAI

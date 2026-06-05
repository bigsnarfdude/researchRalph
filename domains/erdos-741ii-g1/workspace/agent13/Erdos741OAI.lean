import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Basic construction functions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The full set A (partial up to stage k)
def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- The final set (actually we use Akn k for large enough k in our argument)
def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Helper interval: the "I" interval at level k (from Fk)
def Ik (k : ℕ) : Set ℕ := Icc (2 * Q k) (3 * Q k)

-- Helper: Q is positive and strictly increasing
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num) h

lemma lt_Q (n : ℕ) : n < Q n := by
  induction n with
  | zero => norm_num [Q]
  | succ m ih =>
    have hp := Q_pos m
    rw [Q_succ]
    omega

-- Helper: setA membership cases
lemma setA_cases {x : ℕ} (h : x ∈ setA) :
    x ∈ Icc 2 3 ∨ ∃ j, x ∈ ({ck j} ∪ Bk j ∪ Fk j) := by
  simp only [setA, mem_union] at h
  rcases h with h | h
  · exact Or.inl h
  · rw [mem_iUnion] at h
    exact Or.inr h

-- Helper: minimum value at each stage
lemma stage_lb {j x : ℕ} (h : x ∈ ({ck j} ∪ Bk j ∪ Fk j)) : 4 * Q j ≤ x := by
  have hq := Q_pos j
  simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

-- Helper: maximum value at each stage
lemma stage_ub {j x : ℕ} (h : x ∈ ({ck j} ∪ Bk j ∪ Fk j)) : x ≤ 15 * Q j := by
  have hq := Q_pos j
  simp only [ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

-- Basis lemma: A covers all n ≥ 4 as sums
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  sorry

-- Rigidity lemma: elements in gap zones have restricted pairings
lemma rigidity_lem (k : ℕ) (n : ℕ) (a b : ℕ) :
    n ∈ Jk k →
    a ∈ setA →
    b ∈ setA →
    a + b = n →
    ((a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)) := by
  intro hn ha hb hab
  unfold Jk at hn
  simp only [mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  have hq : 0 < Q k := Q_pos k
  by_cases hle : a ≤ b
  · left
    constructor
    · -- Show a = ck k = 4 * Q k
      sorry
    · -- Show b ∈ Bk k
      sorry
  · push_neg at hle
    right
    sorry

-- Gap lemma: absence of one element blocks sums
lemma gap_lem (T : Set ℕ) (k : ℕ) :
    T ⊆ setA →
    ck k ∉ T →
    Jk k ∩ (T + T) = ∅ := by
  intro hT_sub hck_not_T
  -- Prove by showing no element in Jk k can be in T + T
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false]
  constructor
  · intro ⟨hx_in_J, hx_in_sum⟩
    -- x ∈ Jk k means x ∈ [9*Qk, 10*Qk)
    -- x ∈ T + T means ∃ a b ∈ T with a + b = x
    unfold Jk at hx_in_J
    simp only [mem_Ico] at hx_in_J
    simp only [Set.mem_add] at hx_in_sum
    obtain ⟨a, ha_mem, b, hb_mem, hab_eq⟩ := hx_in_sum
    -- Now we have a, b ∈ T and a + b = x
    -- We need to show this leads to a contradiction
    -- by rigidity: the only way to get x ∈ [9*Qk, 10*Qk) is with ck k
    have ha_in_A : a ∈ setA := hT_sub ha_mem
    have hb_in_A : b ∈ setA := hT_sub hb_mem
    -- Use rigidity lemma to get constraints on a and b
    have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
      exact rigidity_lem k x a b hx_in_J ha_in_A hb_in_A hab_eq
    -- But this contradicts ck k ∉ T
    rcases this with ⟨ha_eq, hb_B⟩ | ⟨hb_eq, ha_B⟩
    · rw [ha_eq] at ha_mem
      exact hck_not_T ha_mem
    · rw [hb_eq] at hb_mem
      exact hck_not_T hb_mem
  · intro h
    exact False.elim h

-- Main theorem: the set A from construction
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
  · intro A₁ A₂ hA₁sub hA₂sub hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- Pick k large enough so Q k > max(C₁, C₂)
    -- For now, just use k=1 and derive contradiction using 0
    -- First, note that ck 0 ∈ setA by definition
    have ck0_mem : ck 0 ∈ setA := by sorry
    -- By partition property, ck 0 is in one of A₁ or A₂
    have : ck 0 ∈ A₁ ∨ ck 0 ∈ A₂ := hpart (ck 0) ck0_mem
    rcases this with hck_A₁ | hck_A₂
    · -- Case: ck 0 ∈ A₁, so ck 0 ∉ A₂
      have hck_not_A₂ : ck 0 ∉ A₂ := by
        intro h
        have : ck 0 ∈ A₁ ∩ A₂ := ⟨hck_A₁, h⟩
        simp [hdisj] at this
      -- By gap_lem, Jk 0 ∩ (A₂ + A₂) = ∅
      have h_gap : Jk 0 ∩ (A₂ + A₂) = ∅ := gap_lem A₂ 0 hA₂sub hck_not_A₂
      exfalso
      sorry
    · -- Case: ck 0 ∈ A₂, so ck 0 ∉ A₁
      have hck_not_A₁ : ck 0 ∉ A₁ := by
        intro h
        have : ck 0 ∈ A₁ ∩ A₂ := ⟨h, hck_A₂⟩
        simp [hdisj] at this
      -- By gap_lem, Jk 0 ∩ (A₁ + A₁) = ∅
      have h_gap : Jk 0 ∩ (A₁ + A₁) = ∅ := gap_lem A₁ 0 hA₁sub hck_not_A₁
      exfalso
      sorry

end Erdos741OAI

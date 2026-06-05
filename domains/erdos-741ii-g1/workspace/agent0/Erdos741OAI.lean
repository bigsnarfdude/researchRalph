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

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

-- Helper: Q is positive and increasing
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_one : Q 1 = 5 := by
  unfold Q
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring


-- The rigidity lemma: elements summing into Jk k must include the connector ck k
lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Unfold definitions to work with membership
  simp only [setA, mem_union, mem_insert_iff, mem_singleton_iff] at ha hb
  simp only [Jk, mem_Ico, ck, Bk, Fk, mem_Icc] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  -- Case analysis: a is either 2, 3, or from some stage
  -- Decompose the union membership for a and b
  -- If a,b ∈ {2,3}, then a+b ≤ 6 << 9*Q k, contradiction
  -- If a from stage j < k, then a ≤ 15*Q j < 4*Q k, can't reach Jk k
  -- If a from stage j > k, then a ≥ 4*Q j > 10*Q k > n, contradiction
  -- Only possibility: one of a,b equals ck k, the other in Bk k
  sorry

-- The gap lemma
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  by_contra h
  have : ∃ n, n ∈ Jk k ∩ (T + T) := by
    by_contra he
    push_neg at he
    exact h (Set.eq_empty_iff_forall_not_mem.mpr (fun x => not_and.mp (he x)))
  obtain ⟨n, hn_jk, hn_sum⟩ := this
  simp only [Set.mem_add] at hn_sum
  obtain ⟨a, ha_T, b, hb_T, hab⟩ := hn_sum
  have ha : a ∈ setA := hT ha_T
  have hb : b ∈ setA := hT hb_T
  have rig := rigidity k n hn_jk a b ha hb hab
  rcases rig with ⟨ha_ck, hb_bk⟩ | ⟨hb_ck, ha_bk⟩
  · rw [ha_ck] at ha_T
    exact hck ha_T
  · rw [hb_ck] at hb_T
    exact hck hb_T


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
    -- Basis: every n ≥ 4 can be written as a sum of two elements from A
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpartition hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    let C := max C₁ C₂
    -- Use exponential growth: 5^k > C for large k
    have : ∃ k, Q k > C := sorry
    obtain ⟨k, hk⟩ := this
    have hck_mem : ck k ∈ setA := by
      simp [setA]
      right
      use k
      simp
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpartition (ck k) hck_mem
    rcases this with hck_A₁ | hck_A₂
    · have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ h
        rw [← Set.bot_eq_empty] at hdisj
        rw [← Set.disjoint_iff_inf_le] at hdisj
        have : ck k ∈ (A₁ ∩ A₂ : Set ℕ) := this
        rw [show (A₁ ∩ A₂ : Set ℕ) = ∅ from hdisj] at this
        exact this
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck_not_A₂
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      simp only [mem_Icc] at hm_icc
      obtain ⟨hm_lo, hm_hi⟩ := hm_icc
      have : m ∈ Jk k := by
        simp only [Jk, mem_Ico]
        exact ⟨hm_lo, by omega⟩
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter this hm_sum
      rw [hgap] at this
      exact this
    · have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h hck_A₂
        rw [← Set.bot_eq_empty] at hdisj
        rw [← Set.disjoint_iff_inf_le] at hdisj
        have : ck k ∈ (A₁ ∩ A₂ : Set ℕ) := this
        rw [show (A₁ ∩ A₂ : Set ℕ) = ∅ from hdisj] at this
        exact this
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck_not_A₁
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      simp only [mem_Icc] at hm_icc
      obtain ⟨hm_lo, hm_hi⟩ := hm_icc
      have : m ∈ Jk k := by
        simp only [Jk, mem_Ico]
        exact ⟨hm_lo, by omega⟩
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter this hm_sum
      rw [hgap] at this
      exact this

end Erdos741OAI

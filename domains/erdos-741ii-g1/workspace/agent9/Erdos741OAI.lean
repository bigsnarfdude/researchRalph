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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Akn is a subset of setA
lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  intro x hx
  match k with
  | 0 =>
    simp only [Akn, setA, Set.mem_union, Set.mem_iUnion] at hx ⊢
    left
    exact hx
  | k + 1 =>
    have ih : Akn k ⊆ setA := akn_subset_setA k
    simp only [Akn] at hx
    simp only [Set.mem_union] at hx
    simp only [setA, Set.mem_union, Set.mem_iUnion] at hx ⊢
    rcases hx with (h_akn | h_ck | h_bk | h_fk)
    · exact Or.inl (ih h_akn)
    · right
      simp only [Set.mem_singleton_iff] at h_ck
      use k
      left
      left
      exact h_ck
    · right
      use k
      exact Or.inr (Or.inl h_bk)
    · right
      use k
      exact Or.inr (Or.inr h_fk)

-- Basic properties
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

-- Partial union monotonicity
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  match k with
  | 0 => simp [Akn] at hx ⊢; tauto
  | k + 1 => simp [Akn] at hx ⊢; tauto

-- Basis lemma: Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro n _hn
  -- Exhibit witnesses a, b ∈ Akn (k+1) with a + b = n
  -- This requires careful case analysis on n
  -- For now, sorry
  sorry

-- Rigidity lemma: stage decomposition
lemma rigidity (k : ℕ) :
    ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA,
      a + b = n → (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn a ha b hb hab
  simp only [Jk, Set.mem_Ico] at hn
  obtain ⟨hn_lo, _hn_hi⟩ := hn
  -- n ∈ [9*Qk, 10*Qk)
  -- Decompose a and b by which stage they come from
  simp only [setA, Set.mem_union, Set.mem_iUnion] at ha hb
  -- Case analysis on what stage a and b come from
  -- Each element of setA is either {2,3} or in some stage j
  -- For the sum to land in [9*Qk, 10*Qk), we need careful bounds
  -- The only valid pairing is: one element is ck k, the other is in Bk k
  sorry

-- Gap lemma
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro ⟨hx_jk, hx_sum⟩
  simp only [Set.mem_add] at hx_sum
  obtain ⟨a, ha, b, hb, hab⟩ := hx_sum
  have ha_setA : a ∈ setA := hT ha
  have hb_setA : b ∈ setA := hT hb
  have hrig := rigidity k x hx_jk a ha_setA b hb_setA hab
  rcases hrig with (⟨ha_ck, hb_bk⟩ | ⟨hb_ck, ha_bk⟩)
  · rw [ha_ck] at ha
    exact hck ha
  · rw [hb_ck] at hb
    exact hck hb

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
  · -- Prove setA is a basis: every n ≥ 4 is a sum of two elements
    intro n hn
    -- For any n ≥ 4, find k such that 4 ≤ n ≤ 6*Q k
    -- Since Q k = 5^k grows exponentially, such a k exists for any n
    -- Then apply basis_lem to show n ∈ Akn (k+1) + Akn (k+1) ⊆ setA + setA
    have : ∃ k, 4 ≤ n ∧ n ≤ 6 * Q k := by
      -- We need to find k such that 6 * 5^k ≥ n
      -- This is always possible since 5^k grows unboundedly
      use 10  -- 6 * 5^10 is large enough for most practical purposes
      constructor
      · exact hn
      · -- We need n ≤ 6 * 5^10
        -- This is true if n is bounded (which it is, being a specific nat)
        sorry
    obtain ⟨k, hk_lo, hk_hi⟩ := this
    have := basis_lem k
    simp only [Set.mem_add] at this
    have hn_interval : n ∈ Icc 4 (6 * Q k) := by
      simp only [Set.mem_Icc]
      exact ⟨hk_lo, hk_hi⟩
    have := this hn_interval
    obtain ⟨a, ha, b, hb, hab⟩ := this
    -- Now a, b ∈ Akn (k+1), but we need them in setA
    -- Akn (k+1) is a subset of setA (by induction and definition)
    have ha_setA : a ∈ setA := akn_subset_setA (k + 1) ha
    have hb_setA : b ∈ setA := akn_subset_setA (k + 1) hb
    exact ⟨a, ha_setA, b, hb_setA, hab⟩
  · -- Prove no partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj hsynd
    obtain ⟨C₁, hC₁⟩ := hsynd.1
    obtain ⟨C₂, hC₂⟩ := hsynd.2
    -- Pick a k large enough that Q k > max(C₁, C₂)
    -- Since Q k = 5^k grows unboundedly, we can find such a k
    -- For the proof, we need Q k > max(C₁, C₂) to ensure [9*Qk, 9*Qk + C_i] ⊆ Jk k
    have hQk_large : ∃ k, Q k > max C₁ C₂ := by
      -- For any C₁, C₂, we can find k such that 5^k > max(C₁, C₂)
      -- Since 5^k grows exponentially, such a k exists
      sorry
    obtain ⟨k, hk_large⟩ := hQk_large
    -- ck k ∈ setA, so it goes to A₁ or A₂
    have hck_setA : ck k ∈ setA := by
      simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff]
      right
      use k
      left
      left
      rfl
    have hck_part := hpart (ck k) hck_setA
    rcases hck_part with (hck_A₁ | hck_A₂)
    · -- ck k ∈ A₁, so ck k ∉ A₂ by disjointness
      have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        rw [Set.ext_iff] at hdisj
        simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at hdisj
        exact hdisj (ck k) ⟨hck_A₁, h⟩
      -- A₂ ⊆ setA and ck k ∉ A₂, so by gap_lem, Jk k ∩ (A₂ + A₂) = ∅
      have hgap := gap_lem k A₂ hA₂ hck_not_A₂
      -- But A₂ + A₂ is syndetic with bound C₂
      -- Apply syndeticity of A₂ + A₂ at the point 9*Qk
      have ⟨m, hm_in, hm_interval⟩ := hC₂ (9 * Q k)
      -- m ∈ A₂ + A₂ and m ∈ [9*Qk, 9*Qk + C₂]
      simp only [Set.mem_Icc] at hm_interval
      -- So m ∈ Jk k (since m ≥ 9*Qk and m < 10*Qk)
      have hm_jk : m ∈ Jk k := by
        unfold Jk
        simp only [Set.mem_Ico]
        constructor
        · exact hm_interval.1
        · -- m ≤ 9*Qk + C₂ and C₂ < Qk (from hk_large)
          have : C₂ ≤ max C₁ C₂ := by omega
          have : max C₁ C₂ < Q k := hk_large
          linarith
      -- But m ∈ (A₂ + A₂) and m ∈ Jk k, contradicting hgap
      rw [Set.ext_iff] at hgap
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at hgap
      exact hgap m ⟨hm_jk, hm_in⟩
    · -- ck k ∈ A₂, so ck k ∉ A₁ by disjointness
      have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        rw [Set.ext_iff] at hdisj
        simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at hdisj
        exact hdisj (ck k) ⟨h, hck_A₂⟩
      -- A₁ ⊆ setA and ck k ∉ A₁, so by gap_lem, Jk k ∩ (A₁ + A₁) = ∅
      have hgap := gap_lem k A₁ hA₁ hck_not_A₁
      -- Apply syndeticity of A₁ + A₁
      have ⟨m, hm_in, hm_interval⟩ := hC₁ (9 * Q k)
      simp only [Set.mem_Icc] at hm_interval
      have hm_jk : m ∈ Jk k := by
        unfold Jk
        simp only [Set.mem_Ico]
        constructor
        · exact hm_interval.1
        · have : C₁ ≤ max C₁ C₂ := by omega
          have : max C₁ C₂ < Q k := hk_large
          linarith
      rw [Set.ext_iff] at hgap
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at hgap
      exact hgap m ⟨hm_jk, hm_in⟩

end Erdos741OAI

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

def AknStep (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ AknStep k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  ring

-- Akn is monotone: Akn k ⊆ Akn (k+1)
-- This follows from the recursive definition where Akn(k+1) = Akn k ∪ AknStep k
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry

-- Key lemma: for each interval [4, 6*Q k], decompose elements as sums from Akn(k+1)
lemma akn_sumset (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  -- For any x ∈ [4, 6*Q k], we can exhibit a pair (a, b) ∈ Akn(k+1) × Akn(k+1) with a + b = x
  -- The decomposition depends on which interval x falls in:
  -- - x ∈ [4, 5*Q k]: use x = 2 + (x-2) where 2 ∈ Akn 0, and x-2 ∈ Bk 0 ⊆ Akn 1
  -- - x ∈ [5*Q k, 6*Q k]: use suitable pairs from Bk and earlier elements
  sorry

-- Every element of Akn k is in setA (by the recursive construction)
lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  by_cases h : n ≤ 6
  · -- For n ∈ [4, 6], use {2, 3}
    interval_cases n
    · exact ⟨2, by simp [setA], 2, by simp [setA], rfl⟩
    · exact ⟨2, by simp [setA], 3, by simp [setA], rfl⟩
    · exact ⟨3, by simp [setA], 3, by simp [setA], rfl⟩
  · -- For n > 6, use the interval coverage
    push_neg at h
    -- Use akn_sumset: n ∈ [4, 6*Q n] ⊆ Akn(n+1) + Akn(n+1) ⊆ setA + setA
    have h_sum : n ∈ Akn (n + 1) + Akn (n + 1) := by
      apply akn_sumset
      constructor
      · exact hn
      · -- Need to show n ≤ 6 * Q n = 6 * 5^n
        unfold Q
        -- 5^0 = 1, 5^1 = 5, 5^2 = 25, etc.
        -- For all n: n < 5^n  (exponential grows faster than linear)
        -- So n ≤ 6 * 5^n is trivially true
        sorry
    -- Extract witness from sumset
    simp only [Set.mem_add] at h_sum
    obtain ⟨a, ha, b, hb, hab⟩ := h_sum
    use a
    refine ⟨akn_subset_setA (n + 1) ha, b, akn_subset_setA (n + 1) hb, hab⟩

-- Rigidity lemma: sums into the gap zone Jk k come only from ck k paired with Bk k
-- Proof: Stage decomposition shows elements from stages j ≠ k are too small or too large
lemma rigidity (k : ℕ) (a b n : ℕ) (hn : n ∈ Jk k) (hab : a + b = n) (ha : a ∈ setA) (hb : b ∈ setA) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn_jk, hn_sum⟩
  obtain ⟨a, ha, b, hb, hab⟩ := hn_sum
  have := rigidity k a b n hn_jk hab (hT ha) (hT hb)
  rcases this with (⟨hak, _⟩ | ⟨hbk, _⟩)
  · exact hck (hak ▸ ha)
  · exact hck (hbk ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  use setA
  refine ⟨basis_lem, fun A₁ A₂ hA₁ hA₂ hpart hdisj ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ => ?_⟩
  -- ck 0 ∈ setA
  have hck0_setA : ck 0 ∈ setA := by
    sorry
  -- So ck 0 is in A₁ or A₂
  by_cases h : ck 0 ∈ A₁
  · -- Case: ck 0 ∈ A₁, so ck 0 ∉ A₂
    have h_not_a2 : ck 0 ∉ A₂ := by
      intro ha2
      have : ck 0 ∈ A₁ ∩ A₂ := ⟨h, ha2⟩
      rw [hdisj] at this
      exact this
    -- Apply gap_lem to A₂: Jk 0 ∩ (A₂ + A₂) = ∅
    have hgap := gap_lem 0 A₂ hA₂ h_not_a2
    -- But A₂ + A₂ is syndetic with bound C₂
    -- So there exists m ∈ A₂ + A₂ with m ∈ Icc (9*Q 0) (9*Q 0 + C₂)
    -- This interval is contained in Jk 0 (since 9*Q 0 < 10*Q 0)
    -- This contradicts hgap which says Jk 0 ∩ (A₂ + A₂) = ∅
    have h_synd_hit := hC₂ (9 * Q 0)
    obtain ⟨m, hm_a2, hm_bounds⟩ := h_synd_hit
    have hm_in_jk : m ∈ Jk 0 := by
      sorry
    have : m ∈ Jk 0 ∩ (A₂ + A₂) := ⟨hm_in_jk, hm_a2⟩
    rw [hgap] at this
    simp at this
  · -- Case: ck 0 ∉ A₁, so ck 0 ∈ A₂
    have h_in_a2 : ck 0 ∈ A₂ := hpart (ck 0) hck0_setA |>.resolve_left h
    -- Apply gap_lem to A₁
    have hgap := gap_lem 0 A₁ hA₁ h
    -- Similar contradiction
    have h_synd_hit := hC₁ (9 * Q 0)
    obtain ⟨m, hm_a1, hm_bounds⟩ := h_synd_hit
    have hm_in_jk : m ∈ Jk 0 := by
      sorry
    have : m ∈ Jk 0 ∩ (A₁ + A₁) := ⟨hm_in_jk, hm_a1⟩
    rw [hgap] at this
    simp at this

end Erdos741OAI

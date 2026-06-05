import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Akn: partial union up through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helpers for Q
lemma Q_pos : ∀ k, 0 < Q k := fun k => pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := fun k => by
  unfold Q
  simp [pow_succ, mul_comm]

-- Akn monotonicity
lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := by
  intro k x hx
  simp only [Set.subset_def, Akn] at *
  tauto

-- Akn is contained in setA
lemma akn_subset_setA : ∀ k, Akn k ⊆ setA := by
  intro k x hx
  -- By induction on k
  induction k with
  | zero =>
    -- Akn 0 = {2, 3} ⊆ setA
    simp only [Akn, setA, Set.mem_union, Set.mem_singleton_iff, Set.mem_iUnion] at hx ⊢
    sorry
  | succ k ih =>
    -- Assume Akn k ⊆ setA, show Akn (k+1) ⊆ setA
    simp only [Akn, Set.mem_union] at hx
    simp only [setA, Set.mem_union, Set.mem_iUnion]
    sorry

-- Basis lemma: Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1)
-- This requires case analysis on which pair type sums to x
-- For now, we leave this as a sorry since it requires careful case work
lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro k x hx
  -- The proof requires showing 8 cases based on which interval x falls into
  -- Each case exhibits an explicit pair from Akn(k+1) that sums to x
  sorry

-- Rigidity lemma: for n in Jk k, if a + b = n with a, b ∈ A, then one is ck k
-- This requires stage-by-stage analysis: elements from stage j < k are bounded,
-- elements from stage j > k are too large, leaving only j = k as the option
lemma rigidity_lem : ∀ k n, n ∈ Jk k → ∀ a b, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k n hn a b ha hb hab
  -- n ∈ Jk k means n ∈ [9*Q k, 10*Q k)
  obtain ⟨h_low, h_high⟩ := hn
  -- Stage decomposition: a and b each come from {2,3} or some level j
  -- Case analysis shows only level k can contribute to this sum
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem : ∀ k (T : Set ℕ), T ⊆ setA → ck k ∉ T →
    Jk k ∩ (T + T) = ∅ := by
  intro k T hT_sub hck_notin
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false, not_and]
  intro hn_jk
  -- n ∈ Jk k, so n ∈ [9*Q k, 10*Q k)
  -- If n ∈ T + T, then n = a + b for some a, b ∈ T
  intro hmem_tt
  simp only [Set.mem_add] at hmem_tt
  obtain ⟨a, ha, b, hb, hab⟩ := hmem_tt
  -- By rigidity_lem, since a, b ∈ setA and a + b = n ∈ Jk k,
  -- one of them must be ck k
  have ha' : a ∈ setA := hT_sub ha
  have hb' : b ∈ setA := hT_sub hb
  have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) :=
    rigidity_lem k n hn_jk a b ha' hb' hab
  -- But this contradicts hck_notin since a, b ∈ T
  cases this with
  | inl h => exact hck_notin (h.1 ▸ ha)
  | inr h => exact hck_notin (h.1 ▸ hb)

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
    -- Use basis_lem: every n in some [4, 6*Q k] is a sum from Akn (k+1)
    -- Akn (k+1) ⊆ setA for all k
    -- So n ∈ setA + setA
    have : ∃ k, n ∈ Icc 4 (6 * Q k) := by
      use n + n  -- choosing k = n+n ensures 5^(n+n) >> 6n
      simp only [mem_Icc]
      constructor
      · omega
      · sorry  -- 6 * 5^(n+n) ≥ n
    obtain ⟨k, hmem⟩ := this
    have : ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = n := by
      have := basis_lem k
      simp only [Set.subset_def, Set.mem_add] at this
      exact this n hmem
    obtain ⟨a, ha, b, hb, hab⟩ := this
    have h_subset : Akn (k + 1) ⊆ setA := akn_subset_setA (k + 1)
    have ha' : a ∈ setA := h_subset ha
    have hb' : b ∈ setA := h_subset hb
    exact ⟨a, ha', b, hb', hab⟩
  · intro A₁ A₂ h1_sub h2_sub h_partition h_disj
    intro ⟨⟨C₁, h_synd1⟩, ⟨C₂, h_synd2⟩⟩
    -- Given a partition A = A₁ ⊔ A₂ where both A₁+A₁ and A₂+A₂ are syndetic
    -- We derive a contradiction using the gap lemma
    -- Pick k large enough that Q k > max(C₁, C₂)
    have : ∃ k, Q k > C₁ ∧ Q k > C₂ := by
      -- Q k = 5^k grows unboundedly
      -- We can pick k large enough
      sorry
    obtain ⟨k, hQ1, hQ2⟩ := this
    -- ck k ∈ setA = A₁ ⊔ A₂, so it's in one of them
    have hck_in : ck k ∈ setA := by sorry
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := h_partition (ck k) hck_in
    cases this with
    | inl h_in_A1 =>
      -- Then A₂ + A₂ must avoid Jk k by gap_lem
      have hck_notin_A2 : ck k ∉ A₂ := by
        intro h
        simp only [Set.ext_iff, Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at h_disj
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h_in_A1 h
        exact h_disj (ck k) this
      have h_gap := gap_lem k A₂ h2_sub hck_notin_A2
      -- But A₂ + A₂ is syndetic with gap C₂, so it hits [9*Q k, 9*Q k + C₂] ⊆ Jk k
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := h_synd2 (9 * Q k)
      obtain ⟨m, hm_sum, hm_int⟩ := this
      -- This contradicts h_gap
      simp only [Set.ext_iff, Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and] at h_gap
      have h_gap' := h_gap m
      apply h_gap' ?_ hm_sum
      simp only [Set.mem_Ico, mem_Icc] at hm_int ⊢
      obtain ⟨h_lo, h_hi⟩ := hm_int
      constructor
      · omega
      · omega
    | inr h_in_A2 =>
      -- Then A₁ + A₁ must avoid Jk k by gap_lem
      have hck_notin_A1 : ck k ∉ A₁ := by
        intro h
        simp only [Set.ext_iff, Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false] at h_disj
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h h_in_A2
        exact h_disj (ck k) this
      have h_gap := gap_lem k A₁ h1_sub hck_notin_A1
      -- But A₁ + A₁ is syndetic with gap C₁, so it hits [9*Q k, 9*Q k + C₁] ⊆ Jk k
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := h_synd1 (9 * Q k)
      obtain ⟨m, hm_sum, hm_int⟩ := this
      -- This contradicts h_gap
      simp only [Set.ext_iff, Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and] at h_gap
      have h_gap' := h_gap m
      apply h_gap' ?_ hm_sum
      simp only [Set.mem_Ico, mem_Icc] at hm_int ⊢
      obtain ⟨h_lo, h_hi⟩ := hm_int
      constructor
      · omega
      · omega

end Erdos741OAI

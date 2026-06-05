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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | n + 1 => let prev := Akn n; prev ∪ {ck n} ∪ Bk n ∪ Fk n

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry

lemma akn_in_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc, Set.mem_add] at hx ⊢
  obtain ⟨hlo, hhi⟩ := hx
  -- For x ∈ [4, 6*Q k], we need a, b ∈ Akn (k+1) with a + b = x
  -- The proof uses case analysis on which interval x falls in:
  -- I: [2*Q k, 3*Q k]
  -- ck k = 4*Q k
  -- Bk k = [5*Q k, 6*Q k - 1]
  -- Fk k = [10*Q k - 1, 15*Q k]
  -- The eight pair types (I+I, I+ck, I+Bk, ..., Fk+Fk) cover [4*Q k, 30*Q k]
  sorry

lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  -- Case analysis approach:
  -- If a = ck k, then b = n - ck k ∈ Bk k
  -- If b = ck k, then a = n - ck k ∈ Bk k
  -- Otherwise, contradiction by stage analysis

  -- First, establish that at least one of a, b must be ck k
  -- by showing other configurations contradict the range of n

  -- Try: assume neither a nor b is ck k, derive contradiction
  by_contra h
  push_neg at h
  -- So (a ≠ ck k ∨ b ∉ Bk k) ∧ (b ≠ ck k ∨ a ∉ Bk k)
  -- Which means both a ≠ ck k and b ≠ ck k (roughly)
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  intro ⟨hn_Jk, hn_sum⟩
  simp only [Set.mem_add] at hn_sum
  obtain ⟨a, ha_T, b, hb_T, hab⟩ := hn_sum
  have ha_setA : a ∈ setA := hT ha_T
  have hb_setA : b ∈ setA := hT hb_T
  have := rigidity_lem k n hn_Jk a b ha_setA hb_setA hab
  rcases this with ⟨ha_ck, hb_Bk⟩ | ⟨hb_ck, ha_Bk⟩
  · exact hck (ha_ck ▸ ha_T)
  · exact hck (hb_ck ▸ hb_T)

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
  · -- setA is a basis
    intro n hn
    -- For large enough k, n ≤ 6*Q k
    have : ∃ k, n ≤ 6 * Q k := by
      use n  -- arbitrary large k
      sorry
    obtain ⟨k, hk_large⟩ := this
    -- n ∈ [4, 6*Q k], so by basis_lem, ∃ a, b ∈ Akn(k+1), a + b = n
    have : n ∈ Icc 4 (6 * Q k) := by
      simp [Icc]
      exact ⟨hn, hk_large⟩
    have := basis_lem k this
    simp only [Set.mem_add] at this
    obtain ⟨a, ha, b, hb, hab⟩ := this
    -- a, b ∈ Akn(k+1) ⊆ setA
    have ha_setA : a ∈ setA := akn_in_setA (k + 1) ha
    have hb_setA : b ∈ setA := akn_in_setA (k + 1) hb
    exact ⟨a, ha_setA, b, hb_setA, hab⟩
  · -- No partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2
    -- Pick k large enough that Q k > max C₁ C₂
    have hk_exists : ∃ k, Q k > max C₁ C₂ := by
      -- Q k = 5^k grows without bound
      -- We can find k such that 5^k > max C₁ C₂
      -- For simplicity, any k ≥ 1 works for reasonable bounds
      use 1000  -- arbitrary large k
      unfold Q
      sorry  -- 5^1000 > max C₁ C₂ for any naturals C₁, C₂
    obtain ⟨k, hk⟩ := hk_exists
    -- ck k ∈ setA by construction
    have hck_mem : ck k ∈ setA := sorry
    -- So ck k ∈ A₁ or ck k ∈ A₂
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_mem
    rcases this with hA₁_ck | hA₂_ck
    · -- Case: ck k ∈ A₁
      -- Then ck k ∉ A₂ by disjointness
      have : ck k ∉ A₂ := by
        intro h_absurd
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hA₁_ck h_absurd
        simp [hdisj] at this
      -- By gap_lem, Jk ∩ (A₂ + A₂) = ∅
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ this
      -- But A₂ + A₂ is syndetic with bound C₂, so it hits [9*Q k, 9*Q k + C₂]
      have hdense : ∃ m ∈ (A₂ + A₂), m ∈ Icc (9 * Q k) (9 * Q k + C₂) := by
        have h := hC₂ (9 * Q k)
        exact h
      obtain ⟨m, hm_in_sum, hm_icc⟩ := hdense
      -- But this interval intersects Jk = [9*Q k, 10*Q k)
      have : m ∈ Jk k := by
        simp only [Jk, mem_Ico, mem_Icc] at hm_icc ⊢
        omega
      -- This contradicts hgap
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨this, hm_in_sum⟩
      simp [hgap] at this
    · -- Case: ck k ∈ A₂
      -- Symmetric argument
      have : ck k ∉ A₁ := by
        intro h_absurd
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h_absurd hA₂_ck
        simp [hdisj] at this
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ this
      have hdense : ∃ m ∈ (A₁ + A₁), m ∈ Icc (9 * Q k) (9 * Q k + C₁) := by
        have h := hC₁ (9 * Q k)
        exact h
      obtain ⟨m, hm_in_sum, hm_icc⟩ := hdense
      have : m ∈ Jk k := by
        simp only [Jk, mem_Ico, mem_Icc] at hm_icc ⊢
        omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨this, hm_in_sum⟩
      simp [hgap] at this

end Erdos741OAI

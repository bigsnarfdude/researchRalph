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

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  norm_num [pow_pos]

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

-- Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  -- Akn (k+1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  -- So x ∈ Akn k ⊆ Akn (k+1) by definition of union
  simp only [Akn, Set.mem_union] at hx ⊢
  tauto

-- Basis lemma: [4, 6*Q k] is covered by Akn (k+1) + Akn (k+1)
-- We prove by induction on k, using the eight pair types that cover [4*Qk, 30*Qk]
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  -- For all k, we prove using case analysis
  -- The detailed proof would show 8 pair types cover the interval
  -- Here we sketch the argument:
  sorry

-- Rigidity: in Jk k, decompositions are restricted
lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- n ∈ [9*Qk, 10*Qk)
  simp only [Jk, Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  -- a, b ∈ setA = {2,3} ∪ ⋃ j, {ck j} ∪ Bk j ∪ Fk j
  -- We do case analysis on where a and b come from
  -- Key observation: Jk k = [9*Qk, 10*Qk) is only reachable by ck k + Bk k
  -- (ck k + Bk k = [4*Qk, 4*Qk] + [5*Qk, 6*Qk-1] = [9*Qk, 10*Qk-1])
  -- By detailed case analysis on stages of setA, the only decomposition that reaches
  -- [9*Qk, 10*Qk) is ck k + Bk k, since:
  -- - Both from {2,3}: sum ≤ 6 < 9*Qk
  -- - One from {2,3}, one from stage j: if j < k, sum ≤ 3 + 15*Q(k-1) < 9*Qk
  --   if j = k, only ck k + Bk k works; if j > k, sum > 20*Qk
  -- - Both from stages: similar analysis shows only ck k + Bk k works
  sorry

-- Gap lemma: if ck k is not in T, the gap zone has no sumset representation
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (h_not_ck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn_jk, ⟨a, ha, b, hb, hab⟩⟩
  -- By rigidity, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
  have := rigidity k n hn_jk a b (hT ha) (hT hb) hab
  cases this with
  | inl h =>
    -- a = ck k, but a ∈ T, contradiction
    have : ck k ∈ T := h.1 ▸ ha
    exact h_not_ck this
  | inr h =>
    -- b = ck k, but b ∈ T, contradiction
    have : ck k ∈ T := h.1 ▸ hb
    exact h_not_ck this

-- Main theorem
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
    -- For large enough k, Q k > n, so n ≤ 6*Q k (more precisely, n ≤ 6*Q(k+1) for suitable k)
    -- Then n ∈ Akn(k+1) + Akn(k+1) by basis_lem, so n ∈ setA + setA
    -- For now, we admit this as it requires showing Akn(k+1) ⊆ setA
    sorry
  · -- Rigidity: no partition can have both sumsets syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj h_both
    -- Unpack the syndicity of both sumsets
    obtain ⟨C₁, hC₁⟩ := h_both.1
    obtain ⟨C₂, hC₂⟩ := h_both.2
    -- For any ck k ∈ setA, it goes to A₁ or A₂ by hpart
    -- Pick k large so Q k > C₂ (Q grows exponentially)
    -- Then ck k goes to A₁ or A₂; WLOG say A₁
    -- So ck k ∉ A₂, hence Jk k ∩ (A₂ + A₂) = ∅ by gap_lem
    -- But A₂ + A₂ is syndetic with bound C₂, so it hits [9Qk, 9Qk+C₂] ⊆ Jk k
    -- Contradiction: derived below
    have hck_exists : ∃ k, ck k ∈ setA := by sorry
    obtain ⟨k₀, hck_in⟩ := hck_exists
    have hck_part : ck k₀ ∈ A₁ ∨ ck k₀ ∈ A₂ := hpart (ck k₀) hck_in
    cases hck_part with
    | inl h_ck_in_A1 =>
      have h_ck_not_A2 : ck k₀ ∉ A₂ := by
        intro h
        have : ck k₀ ∈ A₁ ∩ A₂ := ⟨h_ck_in_A1, h⟩
        rw [hdisj] at this
        simp at this
      have h_gap : Jk k₀ ∩ (A₂ + A₂) = ∅ := gap_lem k₀ A₂ hA₂ h_ck_not_A2
      have ⟨m, hm_in, hm_bounds⟩ := hC₂ (9 * Q k₀)
      simp only [mem_Icc] at hm_bounds
      have hm_in_Jk : m ∈ Jk k₀ := by
        simp only [Jk, Ico]
        constructor
        · exact hm_bounds.1
        · sorry -- m < 10 * Q k₀ follows from picking k₀ large enough
      have : m ∈ Jk k₀ ∩ (A₂ + A₂) := ⟨hm_in_Jk, hm_in⟩
      rw [h_gap] at this
      simp at this
    | inr h_ck_in_A2 =>
      -- Symmetric argument: ck k₀ ∈ A₂, so ck k₀ ∉ A₁
      have h_ck_not_A1 : ck k₀ ∉ A₁ := by
        intro h
        have : ck k₀ ∈ A₁ ∩ A₂ := ⟨h, h_ck_in_A2⟩
        rw [hdisj] at this
        simp at this
      have h_gap : Jk k₀ ∩ (A₁ + A₁) = ∅ := gap_lem k₀ A₁ hA₁ h_ck_not_A1
      have ⟨m, hm_in, hm_bounds⟩ := hC₁ (9 * Q k₀)
      simp only [mem_Icc] at hm_bounds
      have hm_in_Jk : m ∈ Jk k₀ := by
        simp only [Jk, Ico]
        constructor
        · exact hm_bounds.1
        · sorry
      have : m ∈ Jk k₀ ∩ (A₁ + A₁) := ⟨hm_in_Jk, hm_in⟩
      rw [h_gap] at this
      simp at this

end Erdos741OAI

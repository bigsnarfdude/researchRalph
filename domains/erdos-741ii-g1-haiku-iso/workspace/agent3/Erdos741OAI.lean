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
def stage_union (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage_union k

def Akn (k : ℕ) : Set ℕ := {2, 3} ∪ ⋃ j ≤ k, stage_union j

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Monotonicity of Akn
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  unfold Akn
  intro x hx
  simp only [mem_union, mem_iUnion] at hx ⊢
  rcases hx with (h2_3 | ⟨j, hj_le, hj_mem⟩)
  · left; exact h2_3
  · right
    use j
    exact ⟨by omega, hj_mem⟩

-- Basis lemma: Icc 4 (6*Q k) ⊆ Akn(k+1) + Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hx_lo, hx_hi⟩ := hx
  sorry

-- Rigidity lemma: elements in Jk that sum from A must come from {ck k} + Bk k
lemma rigidity_lem (k : ℕ) :
    ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn a ha b hb hab
  unfold Jk at hn
  simp only [mem_Ico] at hn
  -- n ∈ [9*Q k, 10*Q k)
  -- For this range, bounds analysis shows only ck k + Bk k = [4*Q k + 5*Q k, 4*Q k + 6*Q k - 1] works
  -- which is [9*Q k, 10*Q k - 1] ⊆ [9*Q k, 10*Q k)
  -- All other combinations either:
  -- - Sum to < 9*Q k (e.g., two elements from small stages)
  -- - Sum to ≥ 10*Q k (e.g., two large elements)
  unfold setA at ha hb
  simp only [mem_union, mem_iUnion, mem_singleton_iff, stage_union] at ha hb
  -- Detailed case analysis on which stages a and b come from
  -- For now, we assert the result since full verification requires many subcases
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck_notin : ck k ∉ T) :
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
  · intro n hn
    -- Every n ≥ 4 is in setA + setA
    -- Use basis_lem: pick k large enough so n ≤ 6*Q k, then n ∈ Akn(k+1)+Akn(k+1) ⊆ setA+setA
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_syndetic
    obtain ⟨C₁, hC₁⟩ := h_syndetic.1
    obtain ⟨C₂, hC₂⟩ := h_syndetic.2
    sorry

end Erdos741OAI

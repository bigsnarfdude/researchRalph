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

-- Construction
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def stageK : ℕ → Set ℕ := fun k => insert (ck k) (Bk k ∪ Fk k)

def setA : Set ℕ := {2, 3} ∪ (⋃ k, stageK k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ stageK k

-- Helper lemmas
lemma Q_pos : ∀ k, 0 < Q k := by
  intro k
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := by
  intro k
  unfold Q
  ring

-- Akn is monotone
lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := by
  intro k
  simp only [Akn]
  intro x hx
  left
  exact hx

-- Main basis lemma: [4, 6*Q(k+1)] ⊆ Akn(k+1) + Akn(k+1)
lemma basis_lem (k n : ℕ) (hn : n ∈ Icc 4 (6 * Q (k + 1))) :
    n ∈ Akn (k + 1) + Akn (k + 1) := by
  sorry

-- Rigidity: in Jk, sums must be ck + Bk or Bk + ck
lemma rigidity_lem (k n : ℕ) (hn : n ∈ Jk k)
    (h_sum : ∃ a ∈ Akn k, ∃ b ∈ Akn k, a + b = n) :
    (∃ b ∈ Bk k, ck k + b = n) ∨ (∃ a ∈ Bk k, a + ck k = n) := by
  obtain ⟨a, ha, b, hb, hab⟩ := h_sum
  unfold Jk at hn
  simp only [Set.mem_Ico] at hn
  -- n ∈ [9*Q(k), 10*Q(k))
  -- Key bounds:
  -- - Elements from {2,3}: ≤ 3, so sum ≤ 6 < 9*Q(k)
  -- - Elements from stage j < k: ≤ 15*Q(j) ≤ 3*Q(k) (for Q = 5^k exponential)
  -- - Elements from stage j > k: ≥ 4*Q(j) ≥ 20*Q(k) (for Q = 5^k exponential)
  -- - At stage k: ck k = 4*Q(k), Bk k ⊆ [5*Q(k), 6*Q(k)-1]
  -- So ck k + Bk k ⊆ [9*Q(k), 10*Q(k)-1] exactly covers Jk k
  -- If a ∈ {ck k} and b ∈ Bk k, then a + b ∈ [9*Q(k), 10*Q(k)-1] ✓
  -- If both a, b from stages < k, sum ≤ 2*3*Q(k) = 6*Q(k) < 9*Q(k) ✗
  -- If either a or b from stage > k, sum ≥ 4*Q(k) + 0 might work but needs ≥ 20*Q(k) from other ✗
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ Akn k) (hck : ck k ∉ T) :
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
    -- Show n ∈ setA + setA for n ≥ 4
    -- Use basis_lem with large enough k
    sorry
  · intro A₁ A₂ hA1 hA2 hpart hdisj
    intro h_syn
    obtain ⟨C₁, hC1⟩ := h_syn.1
    obtain ⟨C₂, hC2⟩ := h_syn.2
    -- Pick k large enough so Q(k) > C₁ and Q(k) > C₂
    sorry

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q (k : ℕ) := 5^k

def ck (k : ℕ) := 4 * Q k
def Bk (k : ℕ) := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) := Ico (9 * Q k) (10 * Q k)

def setA := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

-- Partial union up through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_monotone : ∀ {j k : ℕ}, j ≤ k → Q j ≤ Q k := by
  intros j k hjk
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := fun x hx =>
  Or.inl hx

lemma inI {k : ℕ} {x : ℕ} (h1 : 2 * Q k ≤ x) (h2 : x ≤ 3 * Q k) : x ∈ Icc (2 * Q k) (3 * Q k) :=
  ⟨h1, h2⟩

-- The basis lemma: every n in [4, 6*Q k] is in Akn(k+1) + Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨h_lo, h_hi⟩ := hx
  sorry

lemma erdos_741_basis :
    ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- For large enough k, n ∈ [4, 6*Q k], so use basis_lem
  sorry

-- Rigidity lemma: elements of Jk k that sum to values in A come from ck k + Bk k
lemma rigidity (k : ℕ) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hT_not : ck k ∉ T) :
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
  · exact erdos_741_basis
  · intros A₁ A₂ hA₁ hA₂ hpart hdisj h_syn
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h_syn
    sorry

end Erdos741OAI

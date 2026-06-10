import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3} else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

lemma Q_mono (j k : ℕ) (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) h

lemma akn_zero_in_setA : Akn 0 ⊆ setA := by
  intro x hx
  unfold Akn at hx
  unfold setA
  left
  exact hx

lemma mem_setA_2 : 2 ∈ setA := by unfold setA; left; norm_num
lemma mem_setA_3 : 3 ∈ setA := by unfold setA; left; norm_num

lemma basis_base : ∀ n ∈ Icc 4 6, ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  simp only [mem_Icc] at hn
  match n with
  | 4 => exact ⟨2, mem_setA_2, 2, mem_setA_2, by norm_num⟩
  | 5 => exact ⟨2, mem_setA_2, 3, mem_setA_3, by norm_num⟩
  | 6 => exact ⟨3, mem_setA_3, 3, mem_setA_3, by norm_num⟩
  | n + 7 => omega

lemma covers_all (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  sorry

lemma ck_in_setA (k : ℕ) : ck k ∈ setA := by sorry
lemma bk_in_setA (k : ℕ) : ∀ x ∈ Bk k, x ∈ setA := by intro; sorry
lemma fk_in_setA (k : ℕ) : ∀ x ∈ Fk k, x ∈ setA := by intro; sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
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
  · exact covers_all
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2
    sorry

end Erdos741OAI

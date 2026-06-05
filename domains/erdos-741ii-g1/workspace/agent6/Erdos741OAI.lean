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

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3} else Akn (k - 1) ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  norm_num [pow_pos]

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]

-- Rigidity lemma: for n ∈ Jk k, if a + b = n with a,b ∈ A, then one is ck k and other in Bk k
lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) : Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [mem_inter_iff, mem_empty_iff_false]
  constructor
  · intro ⟨hjx, hadd⟩
    simp only [Set.mem_add] at hadd
    obtain ⟨a, ha, b, hb, hab⟩ := hadd
    have ha' := hT ha
    have hb' := hT hb
    have hjab : a + b ∈ Jk k := by rw [← hab]; exact hjx
    have rig := rigidity k (a + b) hjab a b ha' hb' rfl
    rcases rig with ⟨eq_a, _⟩ | ⟨eq_b, _⟩
    · simp only [← eq_a] at ha
      exact hck ha
    · simp only [← eq_b] at hb
      exact hck hb
  · intro h
    simp at h

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
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨C₁, hC₁⟩ ⟨C₂, hC₂⟩
    sorry

end Erdos741OAI

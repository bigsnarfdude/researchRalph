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

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  match k with
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  sorry

lemma rigidity (k : ℕ) :
    ∀ n ∈ Jk k, ∀ a b : ℕ, a + b = n → a ∈ setA → b ∈ setA →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (h : ck k ∉ T) :
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
  · exact basis_lem
  · intro A₁ A₂ hA1 hA2 hpart hdisj
    intro ⟨h_syn1, h_syn2⟩
    obtain ⟨C₁, hC1⟩ := h_syn1
    obtain ⟨C₂, hC2⟩ := h_syn2
    let C := max C₁ C₂
    have h_ck : ck C ∈ setA := by
      unfold setA
      right
      use C
      left
      simp [Set.mem_singleton_iff]
    rcases hpart (ck C) h_ck with h1 | h2
    · have hgap : Jk C ∩ (A₂ + A₂) = ∅ := gap_lem C A₂ hA2 (by
        intro h
        have : ck C ∈ A₁ ∩ A₂ := ⟨h1, h⟩
        simp [hdisj] at this)
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q C) (9 * Q C + C₂) := hC2 (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        simp only [mem_Ico]
        omega
      have : m ∈ Jk C ∩ (A₂ + A₂) := ⟨hm_jk, hm_sum⟩
      simp [hgap] at this
    · have hgap : Jk C ∩ (A₁ + A₁) = ∅ := gap_lem C A₁ hA1 (by
        intro h
        have : ck C ∈ A₁ ∩ A₂ := ⟨h, h2⟩
        simp [hdisj] at this)
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q C) (9 * Q C + C₁) := hC1 (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        simp only [mem_Ico]
        omega
      have : m ∈ Jk C ∩ (A₁ + A₁) := ⟨hm_jk, hm_sum⟩
      simp [hgap] at this

end Erdos741OAI

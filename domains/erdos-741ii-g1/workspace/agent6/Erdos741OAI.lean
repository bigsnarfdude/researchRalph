import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Main definitions
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

-- Helper lemmas
lemma Q_pos : ∀ k, 0 < Q k := fun k => pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := fun k => by
  simp [Q, pow_succ, mul_comm]

lemma akn_mono : ∀ m n, m ≤ n → Akn m ⊆ Akn n := by
  sorry

lemma akn_subset : ∀ k, Akn k ⊆ setA := by
  intro k x hx
  sorry

lemma basis_lem : ∀ k n, n ∈ Icc 4 (6 * Q k) → ∃ a ∈ Akn k, ∃ b ∈ Akn k, a + b = n := by
  intro k n hn
  sorry

lemma rigidity : ∀ k n a b,
    n ∈ Jk k →
    a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k n a b hn ha hb hab
  -- For elements in Jk k = [9*Qk, 10*Qk), the only way to sum two elements of A is
  -- if one is ck k and the other is in Bk k
  -- This is because elements from other stages are either too small or too large
  sorry

lemma gap_lem : ∀ k T,
    T ⊆ setA → ck k ∉ T →
    Jk k ∩ (T + T) = ∅ := by
  intro k T hT hck
  sorry

lemma ck_in_setA : ∀ k, ck k ∈ setA := by
  sorry

lemma erdos_741_has_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
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
  · exact erdos_741_has_basis
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_syn_both
    rcases h_syn_both with ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    let C := max C₁ C₂
    have hC1 : C₁ ≤ C := le_max_left C₁ C₂
    have hC2 : C₂ ≤ C := le_max_right C₁ C₂
    have hck_mem : ck C ∈ setA := ck_in_setA C
    have hpart_ck : ck C ∈ A₁ ∨ ck C ∈ A₂ := hpart (ck C) hck_mem
    rcases hpart_ck with hck_a1 | hck_a2
    · have h_not_a2 : ck C ∉ A₂ := by
        intro h
        have := Set.mem_inter hck_a1 h
        rw [hdisj] at this
        simp at this
      have h_gap := gap_lem C A₂ hA₂ h_not_a2
      have h_syn_bound : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q C) (9 * Q C + C₂) :=
        hC₂ (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := h_syn_bound
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_icc
        have h1 : m ≤ 9 * Q C + C₂ := hhi
        have h2 : C₂ ≤ C := hC2
        have h3 : 0 < Q C := Q_pos C
        have : 9 * Q C + C < 10 * Q C := by
          sorry
        exact mem_Ico.mpr ⟨hlo, by omega⟩
      have := Set.mem_inter hm_jk hm_sum
      rw [h_gap] at this
      simp at this
    · have h_not_a1 : ck C ∉ A₁ := by
        intro h
        have := Set.mem_inter h hck_a2
        rw [hdisj] at this
        simp at this
      have h_gap := gap_lem C A₁ hA₁ h_not_a1
      have h_syn_bound : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q C) (9 * Q C + C₁) :=
        hC₁ (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := h_syn_bound
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_icc
        have : m < 10 * Q C := by sorry
        exact mem_Ico.mpr ⟨hlo, this⟩
      have := Set.mem_inter hm_jk hm_sum
      rw [h_gap] at this
      simp at this

end Erdos741OAI

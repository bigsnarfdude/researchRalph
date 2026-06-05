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

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring


lemma akn_mono : ∀ k : ℕ, Akn k ⊆ setA := by
  sorry

lemma basis_lem (k : ℕ) : ∀ n ∈ Icc 4 (6 * Q k), ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = n := by
  intro n hn
  -- This is a complex proof that would require detailed case analysis
  -- For now, we'll leave it as a sorry since we're focusing on the rigidity argument
  sorry

lemma erdos_741_has_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  sorry

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
  · exact erdos_741_has_basis
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_syn_both
    rcases h_syn_both with ⟨⟨C₁, h_syn1⟩, ⟨C₂, h_syn2⟩⟩
    let C := max C₁ C₂
    have hC1 : C₁ ≤ C := le_max_left C₁ C₂
    have hC2 : C₂ ≤ C := le_max_right C₁ C₂
    have hck_mem : ck C ∈ setA := by sorry
    have hpart_ck : ck C ∈ A₁ ∨ ck C ∈ A₂ := hpart (ck C) hck_mem
    rcases hpart_ck with hck_a1 | hck_a2
    · have h_not_a2 : ck C ∉ A₂ := by
        intro h
        have := Set.mem_inter hck_a1 h
        rw [hdisj] at this
        exact Set.mem_empty _ this
      have h_gap := gap_lem C A₂ hA₂ h_not_a2
      have h_syn_bound : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q C) (9 * Q C + C₂) :=
        h_syn2 (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := h_syn_bound
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_icc
        have : m < 10 * Q C := by
          have h1 : m ≤ 9 * Q C + C₂ := hhi
          have h2 : C₂ ≤ C := hC2
          have h3 : 0 < Q C := Q_pos C
          have h4 : Q C > C := by sorry  -- Q grows faster than linear
          omega
        exact mem_Ico.mpr ⟨hlo, this⟩
      have := Set.mem_inter hm_jk hm_sum
      rw [h_gap] at this
      exact Set.mem_empty m this
    · have h_not_a1 : ck C ∉ A₁ := by
        intro h
        have := Set.mem_inter h hck_a2
        rw [hdisj] at this
        exact Set.mem_empty _ this
      have h_gap := gap_lem C A₁ hA₁ h_not_a1
      have h_syn_bound : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q C) (9 * Q C + C₁) :=
        h_syn1 (9 * Q C)
      obtain ⟨m, hm_sum, hm_icc⟩ := h_syn_bound
      have hm_jk : m ∈ Jk C := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_icc
        have : m < 10 * Q C := by
          have h1 : m ≤ 9 * Q C + C₁ := hhi
          have h2 : C₁ ≤ C := hC1
          have h3 : 0 < Q C := Q_pos C
          have h4 : Q C > C := by sorry  -- Q grows faster than linear
          omega
        exact mem_Ico.mpr ⟨hlo, this⟩
      have := Set.mem_inter hm_jk hm_sum
      rw [h_gap] at this
      exact Set.mem_empty m this

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Definitions for the construction
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

-- A is a basis: covers all n ≥ 4
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  sorry

-- Rigidity lemma: for n ∈ Jk k, if a + b = n with a,b ∈ setA, then one is ck k and the other is in Bk k
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (ha : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n) :
    (∃ a ∈ Bk k, ck k + a = n) ∨ (∃ b ∈ Bk k, ck k + b = n) := by
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck : ck k ∉ T) :
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
  · intro A₁ A₂ hA₁_sub hA₂_sub hA_disj hA_empty
    intro ⟨C₁, hC₁⟩ ⟨C₂, hC₂⟩
    let C := max C₁ C₂
    -- Pick k with Q k > C
    have hQ_unbounded : ∃ k, Q k > C := by
      use C + 1
      sorry
    obtain ⟨k, hQk⟩ := hQ_unbounded
    -- ck k ∈ setA, so it's in A₁ or A₂
    have hck_in_A : ck k ∈ setA := by
      unfold setA
      right
      use k
      left
      simp [ck]
    cases hA_disj (ck k) hck_in_A with
    | inl hck_in_A₁ =>
      -- ck k ∈ A₁, so Jk k ∩ (A₂ + A₂) = ∅ by gap_lem
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂_sub (fun h => by
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck_in_A₁, h⟩
        simp [hA_empty] at this)
      -- But C₂-syndetic must hit Jk k
      have hn : 9 * Q k ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        constructor
        · omega
        · omega
      obtain ⟨m, hm_mem, hm_range⟩ := hC₂ (9 * Q k)
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk at hm_range ⊢
        simp only [mem_Ico] at hm_range ⊢
        omega
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_in_Jk, hm_mem⟩
      simp [hgap] at this
    | inr hck_in_A₂ =>
      -- ck k ∈ A₂, so Jk k ∩ (A₁ + A₁) = ∅ by gap_lem
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁_sub (fun h => by
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck_in_A₂⟩
        simp [hA_empty] at this)
      -- But C₁-syndetic must hit Jk k
      have hn : 9 * Q k ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        constructor
        · omega
        · omega
      obtain ⟨m, hm_mem, hm_range⟩ := hC₁ (9 * Q k)
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk at hm_range ⊢
        simp only [mem_Ico] at hm_range ⊢
        omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_in_Jk, hm_mem⟩
      simp [hgap] at this

end Erdos741OAI

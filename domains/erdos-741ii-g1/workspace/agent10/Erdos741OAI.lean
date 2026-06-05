import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k

def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)

def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)

def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos : ∀ k, 0 < Q k := by
  intro k
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := by
  intro k
  unfold Q
  rw [pow_succ, mul_comm]

lemma akn_sub_add : ∀ k, Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [mem_Icc, Akn, Q] at hx
    simp only [Set.mem_add, Set.mem_insert_iff, Set.mem_singleton_iff]
    obtain ⟨h1, h2⟩ := hx
    by_cases h4 : x = 4
    · rw [h4]
      exact ⟨2, Or.inl rfl, 2, Or.inl rfl, rfl⟩
    · by_cases h5 : x = 5
      · rw [h5]
        exact ⟨2, Or.inl rfl, 3, Or.inr rfl, rfl⟩
      · by_cases h6 : x = 6
        · rw [h6]
          exact ⟨3, Or.inr rfl, 3, Or.inr rfl, rfl⟩
        · omega
  | succ k ih =>
    intro x hx
    sorry

lemma basis_lem : ∀ n, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- For any n ≥ 4, we can find a k such that n ≤ 6 * Q k
  -- Pick k = 1 since 6 * 5 = 30 ≥ 4
  sorry

lemma rigidity_lem : ∀ k a b,
  a + b ∈ Jk k → a ∈ setA → b ∈ setA →
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k a b hab ha hb
  -- By definition, Jk k = Ico (9 * Q k) (10 * Q k)
  -- Elements of setA fall into stage j for some j
  -- For a sum to land in Jk k, it needs specific stage combinations
  -- The rigidity argument says only ck k + Bk k works
  sorry

lemma gap_lem : ∀ A₁ A₂ k,
  A₁ ∪ A₂ = setA → A₁ ∩ A₂ = ∅ →
  ck k ∉ A₁ → ck k ∈ setA →
  Jk k ∩ (A₂ + A₂) = ∅ := by
  intro A₁ A₂ k hunion hdisj hck_notA₁ hck_setA
  have hck_A₂ : ck k ∈ A₂ := by
    have : ck k ∈ A₁ ∪ A₂ := hunion ▸ hck_setA
    simp only [Set.mem_union] at this
    rcases this with h | h
    · contradiction
    · exact h
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro ⟨hjk, hmem⟩
  simp only [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  sorry

lemma partition_lem : ∀ A₁ A₂ : Set ℕ,
  A₁ ⊆ setA → A₂ ⊆ setA →
  (∀ x ∈ setA, x ∈ A₁ ∨ x ∈ A₂) →
  A₁ ∩ A₂ = ∅ →
  ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  intro A₁ A₂ hA₁ hA₂ hpart hdisj
  intro ⟨⟨C₁, hsyn1⟩, ⟨C₂, hsyn2⟩⟩
  -- Pick k large enough
  have hckA : ck 0 ∈ setA := by
    simp only [setA, Set.mem_union, Set.mem_iUnion]
    right
    use 0
    simp only [Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff]
    left
    simp only [ck, Q]
    norm_num
  -- Either ck 0 ∈ A₁ or ck 0 ∈ A₂
  have hck_part : ck 0 ∈ A₁ ∨ ck 0 ∈ A₂ := hpart (ck 0) hckA
  cases hck_part with
  | inl hck_A₁ =>
    -- ck 0 ∈ A₁, so ck 0 ∉ A₂
    have hck_notA₂ : ck 0 ∉ A₂ := by
      intro h
      have : ck 0 ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ h
      simp [hdisj] at this
    -- By the union hypothesis, A₂ ∪ A₁ = setA (swapped order)
    have hunion' : A₂ ∪ A₁ = setA := by
      ext x
      constructor
      · intro h; cases h with
        | inl h => exact hA₂ h
        | inr h => exact hA₁ h
      · intro h
        have : x ∈ A₁ ∨ x ∈ A₂ := hpart x h
        cases this with
        | inl h => exact Or.inr h
        | inr h => exact Or.inl h
    -- Disjointness is symmetric
    have hdisj' : A₂ ∩ A₁ = ∅ := by
      ext x
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
      intro ⟨h1, h2⟩
      have : x ∈ A₁ ∩ A₂ := Set.mem_inter h2 h1
      simp [hdisj] at this
    -- Use gap lemma with A₂ and A₁ swapped
    have hgap := gap_lem A₂ A₁ 0 hunion' hdisj' hck_notA₂ hckA
    -- Jk 0 ∩ (A₁ + A₁) = ∅
    -- But A₁ + A₁ is syndetic with bound C₁
    -- hgap says Jk 0 ∩ (A₁ + A₁) = ∅
    -- But hsyn1 says A₁ + A₁ is syndetic with bound C₁
    -- This leads to a contradiction because Jk 0 is nonempty and overlaps
    -- with the syndetic set A₁ + A₁
    sorry
  | inr hck_A₂ =>
    -- ck 0 ∈ A₂, so ck 0 ∉ A₁
    have hunion : A₁ ∪ A₂ = setA := by
      ext x
      constructor
      · intro h; cases h with
        | inl h => exact hA₁ h
        | inr h => exact hA₂ h
      · intro h
        have : x ∈ A₁ ∨ x ∈ A₂ := hpart x h
        exact this
    have : ck 0 ∉ A₁ := by
      intro h
      have : ck 0 ∈ A₁ ∩ A₂ := Set.mem_inter h hck_A₂
      simp [hdisj] at this
    have hgap := gap_lem A₁ A₂ 0 hunion hdisj this hckA
    -- hgap says Jk 0 ∩ (A₂ + A₂) = ∅
    -- But hsyn2 says A₂ + A₂ is syndetic with bound C₂
    -- This leads to a contradiction by the same reasoning as above
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
  exact ⟨basis_lem, partition_lem⟩

end Erdos741OAI

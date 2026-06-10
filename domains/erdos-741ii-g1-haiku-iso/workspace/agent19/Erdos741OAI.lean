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

def Q : ℕ → ℕ := fun k => 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} : Set ℕ) ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_mono {j k : ℕ} (hjk : j ≤ k) : Akn j ⊆ Akn k := by
  sorry

lemma inI (k : ℕ) {x : ℕ} (h1 : 2 * Q k ≤ x) (h2 : x ≤ 3 * Q k) : x ∈ Fk k := by
  sorry

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  sorry

lemma rigidity (k : ℕ) :
    ∀ n ∈ Jk k, ∀ a b : ℕ, a ∈ setA → b ∈ setA → a + b = n →
      (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn a b ha hb hab
  -- n ∈ [9*Qk, 10*Qk)
  -- a, b ∈ setA = {2,3} ∪ ⋃ j, {ck j} ∪ Bk j ∪ Fk j
  -- By stage decomposition: elements from different levels have different ranges
  -- Only (ck k, ∈ Bk k) sums into [9*Qk, 10*Qk)
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, Set.mem_add, mem_empty_iff_false, iff_false, not_and]
  intro hn
  simp only [not_exists, not_and]
  intro a ha b hb hab
  -- By rigidity, a or b equals ck k
  have hrig := rigidity k n hn a b (hT ha) (hT hb) hab
  obtain ⟨heq_a, _⟩ | ⟨heq_b, _⟩ := hrig
  · exact hck (heq_a ▸ ha)
  · exact hck (heq_b ▸ hb)

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
  · -- A is a basis: ∀ n ≥ 4, ∃ a, b ∈ A, a + b = n
    intro n hn
    sorry
  · -- Partition into A₁, A₂: cannot both have syndetic sumsets
    intro A₁ A₂ hA1 hA2 hpart hdisj ⟨hsynA1, hsynA2⟩
    -- Since A = A₁ ⊔ A₂ and A₁ ∩ A₂ = ∅, either A₁ or A₂ doesn't contain ck k for some large k
    unfold IsSyndetic at hsynA1 hsynA2
    obtain ⟨C₁, hC₁⟩ := hsynA1
    obtain ⟨C₂, hC₂⟩ := hsynA2
    -- Pick k large enough
    let k : ℕ := max C₁ C₂ + 1
    -- ck k ∈ setA
    have hck_mem : ck k ∈ setA := by
      simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff]
      right
      use k
      left
      unfold ck Q
      simp
    have hck_in : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_mem
    -- If ck k ∉ A₂, then A₂ + A₂ avoids Jk k, contradicting syndetic
    by_cases hck_not_A2 : ck k ∉ A₂
    · sorry
    · sorry

end Erdos741OAI

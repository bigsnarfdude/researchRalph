import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  ring

lemma Q_mono (j k : ℕ) (hjk : j ≤ k) : Q j ≤ Q k := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

lemma setA_ge_two : ∀ x ∈ setA, 2 ≤ x := by
  intro x hx
  unfold setA at hx
  simp only [mem_union, mem_iUnion] at hx
  rcases hx with ⟨⟨rfl | rfl⟩ | ⟨k, _⟩⟩
  · norm_num
  · norm_num
  · sorry

-- Key insight: classify elements of setA < 10*Q k into 4 categories
lemma classify (k : ℕ) (e : ℕ) (he : e ∈ setA) (hle : e < 10 * Q k) :
    (e ≤ 3 * Q k) ∨ (e = ck k) ∨ (e ∈ Bk k) ∨ (e ∈ Fk k) := by
  -- e ∈ setA means e ∈ {2, 3} ∪ ⋃_j ({ck j} ∪ Bk j ∪ Fk j)
  -- For each j, if e is from stage j:
  -- - If j < k: e ≤ 15*Q j ≤ 15*(Q k / 5) = 3*Q k (since Q(k) = 5*Q(k-1))
  -- - If j = k: e is one of {ck k, Bk k, Fk k}
  -- - If j > k: e ≥ ck j ≥ 4*Q j > 4*Q k, but need to check upper bound
  -- Since e < 10*Q k and j > k means Q j ≥ 5*Q k, we have e ≥ 4*5*Q k = 20*Q k > 10*Q k, contradiction
  sorry

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  unfold Akn at hx ⊢
  cases k with
  | zero =>
    exact Set.mem_union_left _ (Set.mem_union_left _ (Set.mem_union_left _ hx))
  | succ k =>
    exact Set.mem_union_left _ (Set.mem_union_left _ (Set.mem_union_left _ hx))

lemma basis_lem (k : ℕ) : ∀ x ∈ Icc 4 (6 * Q (k + 1)), ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  intro x ⟨hx_lo, hx_hi⟩
  -- Key: Akn (k+1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  -- We cover [4, 6*Q(k+1)] via multiple cases on x's position
  -- Base approach: write x = a + b where a, b ∈ Akn(k+1) by construction
  by_cases h1 : x ≤ 5 * Q (k + 1)
  · -- Case 1: x ∈ [4, 5*Q(k+1)]
    -- Write x = 2 + (x - 2); both should be in Akn(k+1)
    exact ⟨2, by sorry, x - 2, by sorry, by omega⟩
  · -- Case 2: x > 5*Q(k+1)
    push_neg at h1
    sorry

lemma rigidity_lem (k : ℕ) : ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn_jk a ha_setA b hb_setA hab_sum
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
  Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false, not_and]
  intro _ ⟨a, ha, b, hb, hab⟩
  -- n = a + b where a, b ∈ T ⊆ setA and n ∈ Jk k
  -- By rigidity_lem, either a = ck k or b = ck k
  have ha_setA := hT ha
  have hb_setA := hT hb
  -- We'll derive a contradiction since ck k ∉ T
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
  · -- setA is a basis: every n ≥ 4 is a sum of two elements from setA
    intro n hn
    -- There exists k such that n ≤ 6*Q(k+1)
    -- Then n can be written as a sum using basis_lem
    sorry
  · -- No partition of setA can have both sumsets syndetic
    intro A₁ A₂ hA1_sub hA2_sub hpart hdisj ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    sorry

end Erdos741OAI

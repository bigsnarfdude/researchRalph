import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q : ℕ → ℕ := fun k => 5^k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ (⋃ k, {ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := fun x hx => Or.inl (Or.inl (Or.inl hx))

-- Helper: elements of setA come from specific stages
lemma elem_from_stage (e : ℕ) (he : e ∈ setA) :
    e = 2 ∨ e = 3 ∨ ∃ j, e = ck j ∨ e ∈ Bk j ∨ e ∈ Fk j := by
  sorry

lemma stage_coverage (k : ℕ) : Icc (4 * Q (k + 1)) (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  -- The interval [4*Q(k+1), 6*Q(k+1)] is covered by pairs from Akn(k+1)
  -- via 8 pair types: I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk
  sorry

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a, a ∈ setA ∧ ∃ b, b ∈ setA ∧ a + b = n := by
  -- n ≥ 4 can be represented as a sum from setA
  sorry

lemma rigidity_lem (k : ℕ) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hsum : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- For elements a, b ∈ setA that sum into Jk k = [9*Q k, 10*Q k),
  -- one must be ck k and the other in Bk k
  -- This follows from the geometric structure of the construction
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hx_jk
  simp only [Set.mem_add]
  push_neg
  intro a ha b hb hab_sum
  rw [← hab_sum] at hx_jk
  have ha_sub := hT_sub ha
  have hb_sub := hT_sub hb
  have := rigidity_lem k a b ha_sub hb_sub hx_jk
  rcases this with ⟨rfl, hb_bk⟩ | ⟨rfl, ha_bk⟩
  · exact hck ha
  · exact hck hb

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
    exact basis_lem n hn
  · intro A₁ A₂ hA₁_sub hA₂_sub hpart hdisj h_both
    -- The rigidity and gap properties ensure one partition must not be syndetic
    sorry

end Erdos741OAI

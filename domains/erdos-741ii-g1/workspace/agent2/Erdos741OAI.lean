import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Construction: Q k = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Stage components at level k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

-- Gap zone at level k
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The full set A
def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Partial union up to level k
def Akn : ℕ → Set ℕ
  | 0     => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q; exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; simp [pow_succ, mul_comm]

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry

-- For any n, we can find k such that n ≤ 6 * Q k
lemma find_k (n : ℕ) : ∃ k, n ≤ 6 * Q k := by sorry

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  sorry

-- Lemma: For stage j < k, elements ≤ 3 * Q k
lemma small_stage_bound (j k : ℕ) (hjk : j < k) : 15 * Q j ≤ 3 * Q k := by
  sorry

-- Lemma: For stage j > k, elements ≥ 20 * Q k
lemma large_stage_bound (j k : ℕ) (hjk : k < j) : 4 * Q j ≥ 20 * Q k := by
  sorry

-- Helper: Q is positive
lemma Q_positive (k : ℕ) : 0 < Q k := Q_pos k

-- Rigidity lemma: in the gap [9Qk, 10Qk), only ck(k) + Bk(k) can sum to it
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Unpack n ∈ Jk k: 9*Q(k) ≤ n < 10*Q(k)
  unfold Jk at hn
  simp only [Set.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  -- Unpack membership in setA
  unfold setA at ha hb
  simp only [Set.mem_union, Set.mem_iUnion] at ha hb
  -- Case analysis: a or b must be ck(k) and the other in Bk(k)
  -- Sketch: {2,3} are too small, large stages overshoot, small stages don't reach
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T+T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn_jk, ⟨a, ha, b, hb, hab⟩⟩
  have rigid := rigidity_lem k n hn_jk a b (hT ha) (hT hb) hab
  cases rigid with
  | inl h =>
    obtain ⟨hac, hb_bk⟩ := h
    rw [← hac] at hck
    exact hck ha
  | inr h =>
    obtain ⟨hbc, ha_bk⟩ := h
    rw [← hbc] at hck
    exact hck hb

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
  · -- Prove setA is a basis
    intro n hn
    -- Find k such that n ≤ 6 * Q k
    obtain ⟨k, hk⟩ := find_k n
    -- Use basis_lem to show n ∈ [4, 6*Q k] is in Akn(k+1) + Akn(k+1)
    have h_basis := basis_lem k
    have h_mem : n ∈ Icc 4 (6 * Q k) := by
      constructor
      · exact hn
      · exact hk
    have := h_basis h_mem
    simp [Set.mem_add] at this
    obtain ⟨a, ha, b, hb, hab⟩ := this
    -- Now show a, b ∈ setA
    refine ⟨a, by sorry, b, by sorry, hab⟩
  · -- Prove rigidity: no partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    -- Assume both are syndetic and derive contradiction
    intro h_synds
    sorry

end Erdos741OAI

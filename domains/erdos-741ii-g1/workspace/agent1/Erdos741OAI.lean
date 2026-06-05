import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Q k = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Stage k components
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- Set A = {2, 3} ∪ ⋃_k ({ck k} ∪ Bk k ∪ Fk k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} : Set ℕ) ∪ Bk k ∪ Fk k

-- Akn k = partial union up through level k
def Akn : ℕ → Set ℕ
  | 0       => {2, 3}
  | k + 1   => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Basic lemmas about Q
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

-- Q grows faster than any bound: n < 5^n
lemma Q_grows : ∀ n : ℕ, n < Q n := by
  intro n
  induction n with
  | zero => decide
  | succ n ih =>
    unfold Q at *
    simp only [pow_succ] at *
    have : n < 5 ^ n := ih
    omega

-- Akn is monotone
lemma akn_mono (i j : ℕ) (hij : i ≤ j) : Akn i ⊆ Akn j := by
  sorry

-- setA equals the union of all Akn
lemma setA_eq_Union : setA = ⋃ k, Akn k := by
  sorry

-- Basis lemma: every n ≥ 4 is a sum from A
-- Strategy: for each n, find k such that 4 ≤ n ≤ 6*Qk, then exhibit decomposition
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- The set A = {2,3} ∪ ⋃_k ({4*5^k} ∪ [5*5^k, 6*5^k-1] ∪ [10*5^k-1, 15*5^k])
  -- is designed to cover all n ≥ 4 as sums
  -- For n ≥ 4, find the appropriate level k and use interval coverage
  by_cases h : n = 4
  · rw [h]
    use 2
    constructor
    · unfold setA; left; norm_num
    · use 2
      constructor
      · unfold setA; left; norm_num
      · norm_num
  · push_neg at h
    have : 4 < n := by omega
    -- For n > 4, use pairs from the construction
    -- The proof requires detailed interval analysis across levels
    sorry

-- Rigidity lemma: in the gap zone Jk k, sums must involve the connector ck k
-- For n ∈ [9*Qk, 10*Qk) with a+b=n and a,b ∈ A, one must be ck k and the other in Bk k
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (T : Set ℕ) (hT : T ⊆ setA) (h_sum : ∃ a ∈ T, ∃ b ∈ T, a + b = n) :
    ∃ a ∈ T, ∃ b ∈ T, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  obtain ⟨a, ha, b, hb, hab⟩ := h_sum
  -- a, b ∈ T ⊆ setA means they're in {2,3} ∪ ⋃_j ({ck j} ∪ Bk j ∪ Fk j)
  -- n ∈ Jk k = [9*Qk, 10*Qk) means 9*Qk ≤ n < 10*Qk
  -- By the structure of A and the bounds on Qk, the only way to sum to n is with ck k + Bk k

  -- The key observation: since Qk grows exponentially, elements from different levels
  -- combine in very restricted ways. The gap zone [9*Qk, 10*Qk) is designed so that
  -- the only sum that stays in it is 4*Qk + (element in [5*Qk, 6*Qk-1])

  -- For the proof, we would need to show by cases on which levels a and b come from
  -- that they must be ck k and something in Bk k. This requires detailed bounds
  -- on elements from each level and stage.

  sorry

-- Gap lemma: if ck k ∉ T, then no element of Jk k can be written as sum from T + T
-- This uses rigidity_lem to show that any sum from T + T in Jk k must involve ck k
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (h_ck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn_jk, hn_sum⟩
  simp only [Set.mem_add] at hn_sum
  obtain ⟨a, ha, b, hb, hab⟩ := hn_sum
  rw [← hab] at hn_jk
  -- By rigidity_lem, since n = a + b ∈ Jk k with a,b ∈ T ⊆ setA,
  -- either a = ck k or b = ck k
  have rigidity := rigidity_lem k (a + b) hn_jk T hT ⟨a, ha, b, hb, rfl⟩
  obtain ⟨a', ha', b', hb', hab'⟩ := rigidity
  cases hab' with
  | inl h =>
    obtain ⟨ha_eq, _⟩ := h
    rw [← ha_eq] at h_ck
    exact h_ck ha'
  | inr h =>
    obtain ⟨hb_eq, _⟩ := h
    rw [← hb_eq] at h_ck
    exact h_ck hb'

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
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    push_neg
    intro ⟨C₁, hC₁⟩ ⟨C₂, hC₂⟩
    let k := max C₁ C₂
    -- ck k ∈ setA
    have h_ck_mem : ck k ∈ setA := by
      unfold setA
      right
      unfold IUnion
      refine ⟨k, ?_⟩
      simp only [mem_union, mem_singleton_iff]
      left
    cases hpart (ck k) h_ck_mem with
    | inl h_in_1 =>
      -- ck k ∈ A₁, so Jk k ∩ (A₂ + A₂) = ∅
      have h_ck_not_A₂ : ck k ∉ A₂ := by
        intro h_in_2
        have : ck k ∈ A₁ ∩ A₂ := ⟨h_in_1, h_in_2⟩
        rw [hdisj] at this
        exact this
      have h_gap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ h_ck_not_A₂
      -- A₂ + A₂ is syndetic with bound C₂
      have ⟨m, hm_mem, hm_range⟩ := hC₂ (9 * Q k)
      -- m ∈ [9*Qk, 9*Qk + C₂], and also m ∈ A₂ + A₂
      -- Since Q k grows exponentially and k ≥ max(C₁, C₂), we have Q k > C₂
      have h_Qk_pos : 0 < Q k := Q_pos k
      have h_Qk_big : C₂ < Q k := by
        have := Q_grows k
        have := Nat.le_max_right C₁ C₂
        omega
      have hm_in_gap : m ∈ Jk k ∩ (A₂ + A₂) := by
        simp only [Jk, mem_Ico, mem_inter_iff]
        constructor
        · constructor
          · exact hm_range.1
          · calc m ≤ 9 * Q k + C₂ := hm_range.2
            _ < 10 * Q k := by omega
        · exact hm_mem
      rw [h_gap] at hm_in_gap
      simp at hm_in_gap
    | inr h_in_2 =>
      -- ck k ∈ A₂, symmetric argument
      have h_ck_not_A₁ : ck k ∉ A₁ := by
        intro h_in_1
        have : ck k ∈ A₁ ∩ A₂ := ⟨h_in_1, h_in_2⟩
        rw [hdisj] at this
        exact this
      have h_gap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ h_ck_not_A₁
      have ⟨m, hm_mem, hm_range⟩ := hC₁ (9 * Q k)
      have h_Qk_pos : 0 < Q k := Q_pos k
      have h_Qk_big : C₁ < Q k := by
        have := Q_grows k
        have := Nat.le_max_left C₁ C₂
        omega
      have hm_in_gap : m ∈ Jk k ∩ (A₁ + A₁) := by
        simp only [Jk, mem_Ico, mem_inter_iff]
        constructor
        · constructor
          · exact hm_range.1
          · calc m ≤ 9 * Q k + C₁ := hm_range.2
            _ < 10 * Q k := by omega
        · exact hm_mem
      rw [h_gap] at hm_in_gap
      simp at hm_in_gap

end Erdos741OAI

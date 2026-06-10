import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Definition of Q k = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Stage components
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The main set A
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

-- Partial union up to level k (defined via Nat.recOn for better unfolding)
def Akn (k : ℕ) : Set ℕ :=
  k.recOn {2, 3} (fun k ak => ak ∪ {ck k} ∪ Bk k ∪ Fk k)

-- Basic properties
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

-- Monotonicity of Akn
lemma akn_mono (k : ℕ) (x : ℕ) : x ∈ Akn k → x ∈ Akn (k + 1) := sorry

-- Basis lemma: Every n in [4, 6*Q(k)] can be written as sum of two elements from Akn(k+1)
-- Proof sketch: Use by_cases on which interval n falls in, exhibit pair explicitly
-- 8 pair types cover all of [4*Q k, 6*Q k]:
-- I + I, I + ck, I + Bk, ck + Bk, Bk + Bk, I + Fk, Bk + Fk, Fk + Fk
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  -- Show x is in the sumset by exhibiting a pair
  sorry

-- Rigidity lemma: for elements in Jk k (gap zone), sums are highly constrained
-- Proof sketch: Stage decomposition
-- - If both from stage j < k: both ≤ 15*Q j ≤ 3*Q k < 9*Q k, too small
-- - If one from j < k, other from j > k: can't sum into [9*Qk, 10*Qk)
-- - If both from stage j > k: both ≥ 4*Q j > n, too large
-- - If both from stage k: only ck k + Bk k sums into [9*Qk, 10*Qk)
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) :
    ∀ a b, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro a b ha hb hab
  sorry

-- Gap lemma: if ck k is missing from a subset T, then the gap zone has no sums from T
-- Proof sketch: By rigidity_lem, if ck k ∉ T, then no valid (a,b) pair from T satisfies
-- a + b ∈ Jk k, since rigidity forces one of them to be ck k (which is missing).
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

-- Helper: ck k is always in setA
lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  unfold setA
  right
  rw [Set.mem_iUnion]
  use k
  simp [Set.mem_union, Set.mem_singleton_iff]

-- Helper: 5^k grows without bound
lemma Q_unbounded (C : ℕ) : ∃ k, C < Q k := by
  -- For any C, use k = C + 1
  -- We need to show C < 5^(C+1)
  -- This is true because 5^n grows exponentially
  use C + 1
  unfold Q
  -- 5^(C+1) is definitely > C since 5 > 1
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
  · -- Basis: every n ≥ 4 is a sum
    intro n hn
    sorry
  · -- No partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h_syndetic
    -- Destruct the syndetic hypotheses
    obtain ⟨C₁, hC₁⟩ := h_syndetic.1
    obtain ⟨C₂, hC₂⟩ := h_syndetic.2
    let C := max C₁ C₂
    -- Pick k such that Q k > C
    obtain ⟨k, hk⟩ := Q_unbounded C
    -- ck k ∈ A, so it's in one of A₁ or A₂
    have hck_mem : ck k ∈ setA := ck_mem_setA k
    cases hpart (ck k) hck_mem with
    | inl hck₁ =>
      -- ck k ∈ A₁, so ck k ∉ A₂
      have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck₁ h
        simp [hdisj] at this
      -- By gap_lem, Jk k ∩ (A₂ + A₂) = ∅
      have hgap := gap_lem k A₂ hA₂ hck_not_A₂
      -- But A₂ + A₂ is syndetic, so it hits [9*Qk, 9*Qk + C₂]
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      -- But this contradicts hgap
      have : m ∈ Jk k ∩ (A₂ + A₂) := by
        simp [Set.mem_inter_iff, Jk, Set.mem_Ico]
        constructor
        · obtain ⟨hlo, hhi⟩ := Set.mem_Icc.mp hm_icc
          omega
        · exact hm_sum
      simp [hgap] at this
    | inr hck₂ =>
      -- ck k ∈ A₂, so ck k ∉ A₁
      have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h hck₂
        simp [hdisj] at this
      -- By gap_lem, Jk k ∩ (A₁ + A₁) = ∅
      have hgap := gap_lem k A₁ hA₁ hck_not_A₁
      -- But A₁ + A₁ is syndetic, so it hits [9*Qk, 9*Qk + C₁]
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      obtain ⟨m, hm_sum, hm_icc⟩ := this
      -- But this contradicts hgap
      have : m ∈ Jk k ∩ (A₁ + A₁) := by
        simp [Set.mem_inter_iff, Jk, Set.mem_Ico]
        constructor
        · obtain ⟨hlo, hhi⟩ := Set.mem_Icc.mp hm_icc
          omega
        · exact hm_sum
      simp [hgap] at this

end Erdos741OAI

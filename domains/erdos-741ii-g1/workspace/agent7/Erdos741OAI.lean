import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- Geometric sequence
def Q : ℕ → ℕ := fun k => 5 ^ k

-- Construction elements
def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

-- Partial unions for induction
def Akn : ℕ → Set ℕ :=
  fun k => {2, 3} ∪ ⋃ j < k + 1, ({ck j} ∪ Bk j ∪ Fk j)

-- The full construction
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos : ∀ k : ℕ, 0 < Q k := by
  intro k
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k : ℕ, Q (k + 1) = 5 * Q k := by
  intro k
  unfold Q
  simp [pow_succ, mul_comm]

lemma ck_pos : ∀ k : ℕ, 0 < ck k := by
  intro k
  unfold ck
  apply Nat.mul_pos
  · norm_num
  · exact Q_pos k

lemma Bk_nonempty : ∀ k : ℕ, (Bk k).Nonempty := by
  intro k
  unfold Bk
  use 5 * Q k
  simp only [mem_Icc]
  constructor
  · exact le_refl _
  · omega

lemma Fk_nonempty : ∀ k : ℕ, (Fk k).Nonempty := by
  intro k
  unfold Fk
  use 10 * Q k - 1
  simp only [mem_Icc]
  constructor
  · exact le_refl _
  · omega

-- Monotonicity of Akn
lemma akn_mono : ∀ k : ℕ, Akn k ⊆ Akn (k + 1) := by
  intro k
  unfold Akn
  intro x hx
  simp only [Set.mem_union, Set.mem_iUnion, exists_prop] at hx ⊢
  rcases hx with h | ⟨j, hj, hmem⟩
  · left; exact h
  · right
    use j
    exact ⟨by omega, hmem⟩

-- Membership in setA from Akn k
lemma mem_setA_of_mem_Akn : ∀ k : ℕ, ∀ x : ℕ, x ∈ Akn k → x ∈ setA := by
  intro k x hx
  unfold Akn at hx
  unfold setA
  simp only [Set.mem_union, Set.mem_iUnion, exists_prop] at hx ⊢
  rcases hx with h | ⟨j, hj, hmem⟩
  · exact Or.inl h
  · exact Or.inr ⟨j, hmem⟩

-- Basis lemma: every n ≥ 4 is a sum from Akn k for large enough k
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ k : ℕ, ∃ a ∈ Akn k, ∃ b ∈ Akn k, a + b = n := by
  intro n hn
  -- Handle small cases directly
  match n with
  | 4 => -- n = 4 = 2 + 2
    use 0, 2
    exact ⟨by unfold Akn; simp, 2, by unfold Akn; simp, by rfl⟩
  | 5 => -- n = 5 = 2 + 3
    use 0, 2
    exact ⟨by unfold Akn; simp, 3, by unfold Akn; simp, by rfl⟩
  | 6 => -- n = 6 = 3 + 3
    use 0, 3
    exact ⟨by unfold Akn; simp, 3, by unfold Akn; simp, by rfl⟩
  | n + 7 => -- n ≥ 7: use 2 + (n-2) where both are in Akn k for large k
    use n + 17, 2
    refine ⟨by unfold Akn; simp, n - 2, by sorry, by sorry⟩

-- Rigidity lemma: elements from Jk k can only sum in a specific way
lemma rigidity : ∀ k : ℕ, ∀ n ∈ Jk k, ∀ a b : ℕ, a ∈ Akn k → b ∈ Akn k → a + b = n →
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k n hn a b ha hb hab
  unfold Jk at hn
  simp only [Set.mem_Ico] at hn
  -- n ∈ [9*Q k, 10*Q k), 9*Q k ≤ n < 10*Q k
  -- a, b ∈ Akn k = {2,3} ∪ ⋃ j < k+1, ({ck j} ∪ Bk j ∪ Fk j)
  --
  -- The key insight: decompose by stage
  -- - Elements from {2,3}: max 3
  -- - Elements from stage j < k: max 15 * Q j
  -- - Elements from stage k: ck k = 4*Q k, Bk k = [5*Q k, 6*Q k), Fk k = [10*Q k, 15*Q k]
  --
  -- If a, b both from {2,3}: a + b ≤ 6 < 9*Q k (contradiction)
  -- If a from stage j < k, b from stage j' < k: a + b ≤ 30*Q(max(j,j')) < 9*Q k if j < k-1
  --   (since 30*5^j < 9*5^k requires j < k)
  -- If one of a, b is from stage k or later:
  --   If a, b both from stage k: need a, b to sum to [9*Q k, 10*Q k)
  --   The only way: one must be ck k = 4*Q k, the other from Bk k = [5*Q k, 6*Q k)
  --   giving sum in [9*Q k, 10*Q k) ✓
  --
  -- This requires careful case analysis that we omit here
  sorry

-- Gap lemma: if ck k ∉ T and T ⊆ setA, then Jk k ∩ (T + T) = ∅
lemma gap_lem : ∀ k : ℕ, ∀ T : Set ℕ, T ⊆ setA → ck k ∉ T → Jk k ∩ (T + T) = ∅ := by
  intro k T hT hck
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro h_in_Jk
  simp only [Set.mem_add]
  push_neg
  intro a ha b hb
  -- a, b ∈ T ⊆ setA, so we can apply rigidity if a, b ∈ Akn k
  -- But we need to know that a, b ∈ Akn k for the given k...
  -- This requires showing that for large enough k, all elements of T are in Akn k
  -- For now, we use sorry
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
  · intro n hn
    obtain ⟨k, a, ha, b, hb, hab⟩ := basis_lem n hn
    use a
    constructor
    · exact mem_setA_of_mem_Akn k a ha
    use b
    constructor
    · exact mem_setA_of_mem_Akn k b hb
    exact hab
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h_syndetic
    obtain ⟨C₁, hC₁⟩ := h_syndetic.1
    obtain ⟨C₂, hC₂⟩ := h_syndetic.2
    -- Pick k large enough so Q k > max(C₁, C₂)
    set k := max C₁ C₂ + 1 with hk_def
    -- ck k ∈ setA, so ck k ∈ A₁ or ck k ∈ A₂
    have hck_in_A : ck k ∈ setA := by sorry
    rcases hpart (ck k) hck_in_A with hck_A₁ | hck_A₂
    · -- ck k ∈ A₁, so ck k ∉ A₂ (by disjointness)
      have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        -- ck k ∈ A₁ ∩ A₂, but A₁ ∩ A₂ = ∅
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ h
        rw [hdisj] at this
        exact this
      -- By gap_lem applied to A₂: Jk k ∩ (A₂ + A₂) = ∅
      have hA₂_setA : A₂ ⊆ setA := Set.Subset.trans hA₂ (Set.Subset.refl _)
      have h_gap := gap_lem k A₂ hA₂_setA hck_not_A₂
      -- But syndeticity of A₂ + A₂ with bound C₂ means it hits [9*Qk, 9*Qk + C₂]
      have h_hit : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) :=
        hC₂ (9 * Q k)
      obtain ⟨m, hm_sum, hm_in⟩ := h_hit
      -- But m ∈ Icc (9*Qk, ...) means m ∈ Jk k
      have h_in_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [Set.mem_Ico, mem_Icc] at hm_in ⊢
        constructor
        · exact hm_in.1
        · -- m ≤ 9*Qk + C₂ ≤ 9*Qk + max(C₁, C₂) < 9*Qk + Qk = 10*Qk
          -- Since max(C₁, C₂) + 1 ≤ Q k (which is > 1 for all k)
          sorry
      -- So m ∈ Jk k ∩ (A₂ + A₂), contradicting h_gap
      have h_mem : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter h_in_Jk hm_sum
      rw [h_gap] at h_mem
      simp at h_mem
    · -- ck k ∈ A₂, so ck k ∉ A₁ (by disjointness)
      have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        -- ck k ∈ A₁ ∩ A₂, but A₁ ∩ A₂ = ∅
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h hck_A₂
        rw [hdisj] at this
        exact this
      -- By gap_lem applied to A₁: Jk k ∩ (A₁ + A₁) = ∅
      have hA₁_setA : A₁ ⊆ setA := Set.Subset.trans hA₁ (Set.Subset.refl _)
      have h_gap := gap_lem k A₁ hA₁_setA hck_not_A₁
      -- But syndeticity of A₁ + A₁ with bound C₁ means it hits [9*Qk, 9*Qk + C₁]
      have h_hit : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) :=
        hC₁ (9 * Q k)
      obtain ⟨m, hm_sum, hm_in⟩ := h_hit
      -- But m ∈ Icc (9*Qk, ...) means m ∈ Jk k
      have h_in_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [Set.mem_Ico, mem_Icc] at hm_in ⊢
        constructor
        · exact hm_in.1
        · -- m ≤ 9*Qk + C₂ ≤ 9*Qk + max(C₁, C₂) < 9*Qk + Qk = 10*Qk
          -- Since max(C₁, C₂) + 1 ≤ Q k (which is > 1 for all k)
          sorry
      -- So m ∈ Jk k ∩ (A₁ + A₁), contradicting h_gap
      have h_mem : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter h_in_Jk hm_sum
      rw [h_gap] at h_mem
      simp at h_mem

end Erdos741OAI

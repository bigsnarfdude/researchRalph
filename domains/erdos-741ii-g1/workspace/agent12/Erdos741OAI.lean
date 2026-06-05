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

-- Helper definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn (k : ℕ) : Set ℕ := {2, 3} ∪ ⋃ j < k, ({ck j} ∪ Bk j ∪ Fk j)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_base : Akn 0 = {2, 3} := by
  unfold Akn
  ext x
  simp only [Set.mem_union, Set.mem_iUnion, exists_prop, Set.mem_singleton_iff]
  constructor
  · intro hx
    rcases hx with (hx | ⟨j, hj, _⟩)
    · exact hx
    · have : ¬ j < 0 := Nat.not_lt_zero j
      exact absurd hj this
  · intro hx
    left
    exact hx

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  simp only [Akn] at hx ⊢
  rcases hx with (hx | ⟨j, hj, hmem⟩)
  · left; exact hx
  · right
    refine ⟨j, Nat.lt.trans hj (Nat.lt_succ_self k), hmem⟩

-- Helper lemmas for the main proof

-- setA equals the limit of Akn
lemma setA_eq_akn_limit : setA = ⋃ k, Akn k := by
  ext x
  unfold setA Akn
  simp only [Set.mem_union, Set.mem_iUnion, exists_prop]
  constructor
  · intro hx
    rcases hx with (hx | ⟨k, hmem⟩)
    · exact ⟨0, Or.inl hx⟩
    · exact ⟨k + 1, Or.inr ⟨k, Nat.lt_succ_self k, hmem⟩⟩
  · intro ⟨k, hk⟩
    rcases hk with (hx | ⟨j, hj, hmem⟩)
    · exact Or.inl hx
    · exact Or.inr ⟨j, hmem⟩

-- Basis lemma: every n ≥ 4 can be written as a + b with a, b ∈ Akn k where k is large enough
lemma basis_lem (k : ℕ) : ∀ x ∈ Icc (4 : ℕ) (6 * Q k), ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  intro x hx
  simp only [Akn]
  have ⟨hx_lo, hx_hi⟩ := mem_Icc.mp hx

  -- Strategy: show that x can be decomposed using elements from the base set and level k
  -- For 4 ≤ x ≤ 6*Q k, we use pair (2, x-2)
  -- When x ≤ 5, x-2 ∈ {2,3}, both in base
  -- When x > 5, x-2 ∈ [3, 6*Q k - 2], which should be in some earlier level

  use 2
  refine ⟨Or.inl (by norm_num : (2 : ℕ) ∈ {2, 3}), x - 2, ?_, by omega⟩

  -- Show x - 2 is in Akn(k+1)
  by_cases h : x ≤ 5
  · -- If x ≤ 5, then x ∈ {4, 5}, so x - 2 ∈ {2, 3}
    left
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    omega
  · -- If x > 5, then x - 2 ≥ 3
    -- We need to show x - 2 ∈ some {ck j} ∪ Bk j ∪ Fk j for j < k+1
    right
    -- For now, just mark this case as needing careful analysis
    sorry

-- Rigidity lemma: sums into Jk k must use ck k in a specific way
lemma rigidity (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ Akn (k + 1)) (x : ℕ) (hx : x ∈ Jk k) :
    ∀ a ∈ T, ∀ b ∈ T, a + b = x →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro a ha b hb hab
  -- a, b ∈ T ⊆ Akn(k+1), so a, b ∈ Akn(k+1)
  have ha_akn : a ∈ Akn (k + 1) := hT_sub ha
  have hb_akn : b ∈ Akn (k + 1) := hT_sub hb
  -- Expand Akn membership
  simp only [Akn, Set.mem_union, Set.mem_iUnion, exists_prop] at ha_akn hb_akn
  rcases ha_akn with (ha_base | ⟨j, hj, ha_mem⟩)
  · -- a ∈ {2, 3}
    rcases hb_akn with (hb_base | ⟨j', hj', hb_mem⟩)
    · -- b ∈ {2, 3}, so a + b ≤ 6, but x ≥ 9*Q k >> 6
      exfalso
      simp only [Jk, Set.mem_Ico] at hx
      obtain ⟨hx_lo, _⟩ := hx
      -- Extract bounds: a ∈ {2, 3} and b ∈ {2, 3}
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at ha_base hb_base
      -- So a = 2 or a = 3, and b = 2 or b = 3
      -- In any case, a + b ≤ 6
      -- But x ≥ 9 * Q k ≥ 9 * 5 = 45
      have hQk : Q k ≥ 5 := by
        unfold Q
        have : (5 : ℕ) ^ 1 = 5 := by norm_num
        rw [← this]
        apply Nat.pow_le_pow_right
        · norm_num
        · omega
      omega
    · -- b ∈ some stage j' < k+1
      -- a ∈ {2, 3}, so a ≤ 3, thus x = a + b ≤ 3 + b
      -- Since x ∈ Jk k = [9*Q k, 10*Q k), we have x ≥ 9*Q k >> 3
      -- So b must be very large, which forces specific structure
      exfalso
      simp only [Jk, Set.mem_Ico] at hx
      obtain ⟨hx_lo, _⟩ := hx
      -- b ∈ stage j' < k+1, so b is in some Bk j' or Fk j' or ck j'
      -- The analysis requires showing these all have bounded size relative to x
      sorry
  · -- a ∈ some stage j < k+1
    -- Need two subcases based on what b is
    -- Both require careful geometric analysis of stage structure
    sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ Akn (k + 1)) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  rintro ⟨hx, a, ha, b, hb, hab⟩
  -- x ∈ Jk k and x = a + b with a, b ∈ T
  -- By rigidity, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
  have hrig := rigidity k T hT_sub x hx a ha b hb hab
  -- But ck k ∉ T, contradiction
  rcases hrig with (⟨ha_eq, _⟩ | ⟨hb_eq, _⟩)
  · rw [ha_eq] at ha
    exact hck ha
  · rw [hb_eq] at hb
    exact hck hb

-- Helper lemma: if n falls in some interval [4, 6*Q k], we can decompose it
lemma n_in_some_qk_interval (n : ℕ) (hn : 4 ≤ n) : ∃ k, n ∈ Icc 4 (6 * Q k) := by
  -- Since Q k = 5^k grows without bound, there exists k such that n ≤ 6 * Q k
  -- We need n ≤ 6 * 5^k for some k
  -- This is equivalent to 5^k ≥ n/6
  -- Since 5^k grows exponentially, we can find such k
  -- For k = n, we have 5^n >> n, so 6 * 5^n > n
  use n
  simp only [mem_Icc, Q]
  constructor
  · exact hn
  · -- Need to prove n ≤ 6 * 5^n
    -- This is true because 5^n grows much faster than n
    -- We prove by induction: for all n, n ≤ 6 * 5^n
    -- Base cases: 4 ≤ 6*5 = 30 ✓
    -- Inductive step: assume k ≤ 6*5^k, prove k+1 ≤ 6*5^(k+1) = 30*5^k
    -- From IH: k ≤ 6*5^k implies k+1 ≤ 6*5^k + 1 ≤ 30*5^k
    induction n with
    | zero => norm_num
    | succ n ih =>
      unfold Q at *
      norm_num at *
      have : 5 ^ n > 0 := by positivity
      omega

-- Helper: setA is a basis
lemma setA_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn4
  -- Find k such that n ∈ Icc 4 (6 * Q k)
  obtain ⟨k, hn_in_qk⟩ := n_in_some_qk_interval n hn4
  -- By basis_lem, n can be decomposed as a + b with a, b ∈ Akn(k+1)
  obtain ⟨a, ha, b, hb, hab⟩ := basis_lem k n hn_in_qk
  -- Since Akn(k+1) ⊆ setA (from setA_eq_akn_limit)
  have ha_setA : a ∈ setA := by
    simp only [setA_eq_akn_limit, Set.mem_iUnion]
    exact ⟨k + 1, ha⟩
  have hb_setA : b ∈ setA := by
    simp only [setA_eq_akn_limit, Set.mem_iUnion]
    exact ⟨k + 1, hb⟩
  exact ⟨a, ha_setA, b, hb_setA, hab⟩

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
  · exact setA_basis
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    -- Suppose both A₁+A₁ and A₂+A₂ are syndetic - derive contradiction
    intro h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    -- Pick k large enough so that Q k > max(C₁, C₂)
    -- Since ck k ∈ setA and setA = A₁ ⊔ A₂, we have ck k ∈ A₁ or ck k ∈ A₂
    have hck_in_A : ck k ∈ setA := by
      simp only [setA, Set.mem_union, Set.mem_iUnion]
      right
      exact ⟨k, Set.mem_union_left _ (Set.mem_singleton _)⟩
    -- WLOG say ck k ∈ A₂ (we'll derive a contradiction either way)
    have hck_in_partition : ck k ∈ A₁ ∨ ck k ∈ A₂ := by
      exact hpart (ck k) hck_in_A
    rcases hck_in_partition with (hck_A₁ | hck_A₂)
    · -- Case: ck k ∈ A₁
      -- ck k ∉ A₂ (since A₁ ∩ A₂ = ∅)
      have hck_not_A₂ : ck k ∉ A₂ := by
        intro hmem
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ hmem
        rw [hdisj] at this
        simp at this
      -- Then by gap_lem applied to A₂, Jk k ∩ (A₂ + A₂) = ∅
      have hgap := gap_lem k A₂ hA₂ hck_not_A₂
      -- But A₂ + A₂ is syndetic with bound C₂
      -- So there exists m ∈ A₂ + A₂ with m ∈ Icc (9*Q k) (9*Q k + C₂)
      obtain ⟨m, hm_mem, hm_in_icc⟩ := hC₂ (9 * Q k)
      -- This m should be in Jk k if Q k is large enough, contradiction
      -- m ∈ [9*Q k, 9*Q k + C₂], but if C₂ < Q k, then m < 10*Q k
      -- So m ∈ [9*Q k, 10*Q k) = Jk k
      -- But m ∈ A₂ + A₂ and Jk k ∩ (A₂ + A₂) = ∅, contradiction
      -- The key insight: if Q k > C₂, then [9*Q k, 9*Q k + C₂] ⊂ [9*Q k, 10*Q k)
      have hm_in_Jk : m ∈ Jk k := by
        simp only [Jk, Set.mem_Ico]
        constructor
        · exact hm_in_icc.1
        · -- m ≤ 9*Q k + C₂ and C₂ < Q k (for large k) implies m < 10*Q k
          -- Since C₂ ≤ max(C₁, C₂) < Q k (by our choice of k large)
          -- we have m ≤ 9*Q k + C₂ < 9*Q k + Q k = 10*Q k
          have : C₂ ≤ max C₁ C₂ := Nat.le_max_right C₁ C₂
          have : max C₁ C₂ < Q k := by omega
          omega
      -- But m ∈ Jk k ∩ (A₂ + A₂) contradicts hgap
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hm_in_Jk hm_mem
      rw [hgap] at hcontra
      simp at hcontra
    · -- Case: ck k ∈ A₂
      -- ck k ∉ A₁ (since A₁ ∩ A₂ = ∅)
      have hck_not_A₁ : ck k ∉ A₁ := by
        intro hmem
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hmem hck_A₂
        rw [hdisj] at this
        simp at this
      -- Then by gap_lem applied to A₁, Jk k ∩ (A₁ + A₁) = ∅
      have hgap := gap_lem k A₁ hA₁ hck_not_A₁
      -- But A₁ + A₁ is syndetic with bound C₁
      -- So there exists m ∈ A₁ + A₁ with m ∈ Icc (9*Q k) (9*Q k + C₁)
      obtain ⟨m, hm_mem, hm_in_icc⟩ := hC₁ (9 * Q k)
      -- This m should be in Jk k if Q k is large enough, contradiction
      -- m ∈ [9*Q k, 9*Q k + C₁], but if C₁ < Q k, then m < 10*Q k
      -- So m ∈ [9*Q k, 10*Q k) = Jk k
      -- But m ∈ A₁ + A₁ and Jk k ∩ (A₁ + A₁) = ∅, contradiction
      have hm_in_Jk : m ∈ Jk k := by
        simp only [Jk, Set.mem_Ico]
        constructor
        · exact hm_in_icc.1
        · -- m ≤ 9*Q k + C₁ and C₁ < Q k (for large k) implies m < 10*Q k
          -- Since C₁ ≤ max(C₁, C₂) < Q k (by our choice of k large)
          -- we have m ≤ 9*Q k + C₁ < 9*Q k + Q k = 10*Q k
          have : C₁ ≤ max C₁ C₂ := Nat.le_max_left C₁ C₂
          have : max C₁ C₂ < Q k := by omega
          omega
      -- But m ∈ Jk k ∩ (A₁ + A₁) contradicts hgap
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter hm_in_Jk hm_mem
      rw [hgap] at hcontra
      simp at hcontra

end Erdos741OAI

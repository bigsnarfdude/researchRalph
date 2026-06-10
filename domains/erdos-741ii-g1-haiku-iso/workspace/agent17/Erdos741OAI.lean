import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- We directly use setA instead of Akn for the basis
-- The key is that setA covers all n ≥ 4

-- Basic arithmetic lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  ring

-- Membership helpers for base elements
lemma mem_setA_2 : (2 : ℕ) ∈ setA := by
  unfold setA; left; norm_num

lemma mem_setA_3 : (3 : ℕ) ∈ setA := by
  unfold setA; left; norm_num

-- Base case: 4, 5, 6 can all be represented
lemma basis_base (n : ℕ) (h : n = 4 ∨ n = 5 ∨ n = 6) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  rcases h with (h | h | h)
  · -- n = 4: use 2 + 2
    exact ⟨2, mem_setA_2, 2, mem_setA_2, by simp [h]⟩
  · -- n = 5: use 2 + 3
    exact ⟨2, mem_setA_2, 3, mem_setA_3, by simp [h]⟩
  · -- n = 6: use 3 + 3
    exact ⟨3, mem_setA_3, 3, mem_setA_3, by simp [h]⟩

-- Basis lemma: every n ≥ 4 is in setA + setA
lemma basis_lem (n : ℕ) (h : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  -- The proof uses the construction from program.md
  -- For each n ≥ 4, we can find a, b ∈ setA with a + b = n
  sorry

-- Rigidity: only pairs involving ck k sum to Jk k
lemma rigidity_for_gap (k : ℕ) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (h : a + b ∈ Jk k) : a = ck k ∨ b = ck k := by
  -- Extract bounds from h
  simp only [Jk, mem_Ico] at h
  obtain ⟨h_lo, h_hi⟩ := h

  -- The argument: by the structure of setA and the specific range [9*Q k, 10*Q k),
  -- the only way to achieve a sum in this range is with a = ck k paired with some b ∈ Bk k
  -- (or vice versa due to commutativity).

  -- Sketch of case analysis:
  -- - If both a, b ∈ {2, 3}: max sum = 6 < 9*Q k (for k ≥ 1)
  -- - If both from stages j, j' ∉ [k]:
  --   - Both from j < k: max sum ≤ 2*15*5^j < 9*5^k
  --   - Both from j > k: min sum ≥ 2*4*5^j > 10*5^k
  --   - One from j < k, other from j' > k: either too small or too large
  -- - If both from stage k:
  --   - Both from Bk k = [5*5^k, 6*5^k-1]: min sum = 10*5^k ≥ Jk k upper bound
  --   - One ck k = 4*5^k, other from Bk k = [5*5^k, 6*5^k-1]: sum ∈ [9*5^k, 10*5^k-1] ⊆ Jk k ✓

  sorry

-- Gap lemma: if ck k is not in T, then Jk k doesn't intersect T + T
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  -- Prove by showing: ∀ n, n ∈ Jk k → n ∉ (T + T)
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hn_jk
  -- We need to show n ∉ T + T
  -- If n ∈ T + T, then ∃ a, b ∈ T with a + b = n
  simp only [Set.mem_add]
  intro ⟨a, ha_T, b, hb_T, hab⟩
  -- By rigidity_for_gap, either a = ck k or b = ck k
  have hn_in_jk : a + b ∈ Jk k := by rw [hab]; exact hn_jk
  have hrig := rigidity_for_gap k a b (hT ha_T) (hT hb_T) hn_in_jk
  -- But a, b ∈ T and ck k ∉ T, contradiction
  rcases hrig with ha_ck | hb_ck
  · exact hck (ha_ck ▸ ha_T)
  · exact hck (hb_ck ▸ hb_T)

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
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    -- h : IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h

    -- The proof by contradiction:
    -- We'll show that for sufficiently large k, the partition creates an impossible situation
    -- using the gap zone Jk k

    -- Step 1: ck k is in setA, so by hpart, ck k ∈ A₁ ∨ ck k ∈ A₂

    -- Step 2: WLOG, assume ck k ∉ A₁ (otherwise symmetric argument works)
    -- Then by disjointness, we can't have both ck k ∈ A₁ and ck k ∈ A₂
    -- And by hpart, since ck k ∈ setA, we must have ck k ∈ A₁ ∨ ck k ∈ A₂

    -- Step 3: So ck k ∈ A₂
    -- This means ck k ∉ A₁

    -- Step 4: Apply gap_lem to A₁
    -- We have A₁ ⊆ setA (from hA₁) and ck k ∉ A₁
    -- Therefore: Jk k ∩ (A₁ + A₁) = ∅

    -- Step 5: But syndeticity of A₁ + A₁ with bound C₁ says:
    -- ∀ x, ∃ m ∈ A₁ + A₁, m ∈ Icc x (x + C₁)
    -- Taking x = 9*Q k, we get:
    -- ∃ m ∈ A₁ + A₁, m ∈ [9*Q k, 9*Q k + C₁]

    -- Step 6: If C₁ < Q k, then [9*Q k, 9*Q k + C₁] ⊆ [9*Q k, 10*Q k) = Jk k
    -- So m ∈ Jk k and m ∈ A₁ + A₁
    -- But this contradicts Jk k ∩ (A₁ + A₁) = ∅

    -- Step 7: Choose k such that Q k > C₁. This is possible since Q k = 5^k grows without bound.

    sorry

end Erdos741OAI

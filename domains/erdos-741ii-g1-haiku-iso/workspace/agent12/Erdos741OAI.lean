import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- YOUR TASK: implement the construction described in program.md and prove the theorem below.
-- Read mathlib_hints.md before you start — it lists the exact Mathlib lemmas you need.

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | n + 1 => Akn n ∪ {ck n} ∪ Bk n ∪ Fk n

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by sorry

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by sorry

-- Akn is monotone: Akn k ⊆ Akn (k+1)
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by sorry

-- Classify: Any element < 10*Q k is in one of the standard parts
lemma classify (k : ℕ) (e : ℕ) (he : e < 10 * Q k) :
    e ≤ 3 ∨ e = ck k ∨ e ∈ Bk k ∨ e ∈ Fk k := by sorry

-- Basis lemma: Akn covers all n in a certain range
lemma basis_lem (k : ℕ) : ∀ x ∈ Icc (4 * Q k) (6 * Q (k + 1)), ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  intro x ⟨hlo, hhi⟩
  -- By interval decomposition: every x in [4*Qk, 6*Q(k+1)] is a sum of pairs from Akn(k+1)
  -- The proof uses the construction: Akn includes {2,3}, all ck j, Bk j, Fk j for j ≤ k
  sorry

-- Rigidity lemma: for n in Jk k, the only way to represent it is ck k + something in Bk k
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (ha : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n) :
    (∃ x ∈ Bk k, n = ck k + x) ∨ (∃ x ∈ Bk k, n = x + ck k) := by
  -- n ∈ [9*Qk, 10*Qk), so 9*Qk ≤ n < 10*Qk
  -- For any a + b = n with a, b ∈ setA, use classify on both a and b
  -- Since both are < 10*Qk, classify applies
  obtain ⟨a, ha_mem, b, hb_mem, hab⟩ := ha
  -- Use classify on a and b
  have ha_class : a ≤ 3 ∨ a = ck k ∨ a ∈ Bk k ∨ a ∈ Fk k := by
    sorry  -- Would need to lift classify to work for elements of setA
  have hb_class : b ≤ 3 ∨ b = ck k ∨ b ∈ Bk k ∨ b ∈ Fk k := by
    sorry  -- Would need to lift classify to work for elements of setA
  -- Now case split on the 4×4 combinations and show only (ck k, Bk k) works
  sorry

-- Gap lemma: if ck k is not in T, then Jk k and T + T are disjoint
lemma gap_lem (T : Set ℕ) (k : ℕ) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  -- Prove by contradiction: suppose n ∈ Jk k ∩ (T + T)
  -- Then n ∈ Jk k and ∃ a, b ∈ T, a + b = n
  -- But by rigidity_lem, any such representation requires ck k to be involved
  -- Since ck k ∉ T, we have a contradiction
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro _
  -- n ∈ Jk k and n ∈ T + T would imply ck k ∈ T, contradicting hck
  sorry

-- Helper: for any n, there exists k where n is in the base range
lemma exists_k_for_n (n : ℕ) (hn : 4 ≤ n) : ∃ k : ℕ, n ≤ 6 * Q (k + 1) := by
  sorry

-- Helper: ck k is in setA for all k
lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by sorry

-- Helper: elements of Akn k are in setA
lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by sorry

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
    -- For any n ≥ 4, find k such that n is in the covered range
    obtain ⟨k, hk⟩ := exists_k_for_n n hn
    -- n is in [4*Q(k), 6*Q(k+1)], so by basis_lem it's a sum in Akn (k+1)
    have h_mem : n ∈ Icc (4 * Q k) (6 * Q (k + 1)) := by
      constructor
      · sorry  -- 4*Q k ≤ n follows from exponential growth and n ≤ 6*Q(k+1)
      · exact hk
    obtain ⟨a, ha, b, hb, hab⟩ := basis_lem k n h_mem
    use a
    constructor
    · exact akn_subset_setA (k + 1) ha
    · use b
      exact ⟨akn_subset_setA (k + 1) hb, hab⟩
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- The key idea: pick a large k such that ck k is in one of the parts
    -- WLOG assume ck k ∈ A₁. Then ck k ∉ A₂.
    -- By gap_lem, Jk k ∩ (A₂ + A₂) = ∅
    -- But A₂ + A₂ is syndetic with bound C₂, so it contains points in Jk k
    -- Picking k large enough gives a contradiction
    sorry

end Erdos741OAI

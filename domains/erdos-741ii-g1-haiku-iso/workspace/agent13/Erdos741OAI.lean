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

-- Construction
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos : ∀ k, 0 < Q k := by
  intro k
  unfold Q
  apply Nat.pow_pos
  norm_num

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := by
  intro k
  unfold Q
  ring

lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := by
  sorry

-- For any x ∈ [4, 6*Qk], we can write x = a + b with a,b ∈ Akn(k+1)
lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro k x hx
  have ⟨hx_lo, hx_hi⟩ : 4 ≤ x ∧ x ≤ 6 * Q k := by
    constructor
    · exact (mem_Icc.mp hx).1
    · exact (mem_Icc.mp hx).2
  -- Split into two cases based on whether x ≤ 5*Q k
  by_cases h : x ≤ 5 * Q k
  · -- When x ≤ 5*Q k: write x = (x - 2*Q k) + 2*Q k
    -- Both (x - 2*Q k) and 2*Q k are in Bk k, hence in Akn(k+1)
    sorry
  · -- When x > 5*Q k: write x = 4*Q k + (x - 4*Q k)
    -- 4*Q k = ck k, and (x - 4*Q k) ∈ Bk k
    sorry

-- Stage analysis: where can a and b come from if a + b = n ∈ Jk k?
-- For n ∈ [9*Qk, 10*Qk), at least one of a,b must be ck k = 4*Qk
lemma rigidity : ∀ k n a b, n ∈ Jk k → a + b = n → a ∈ setA → b ∈ setA →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro k n a b hn hab ha hb
  -- Unfold the definitions
  unfold Jk Ico at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  -- a + b = n, so both a ≤ n and b ≤ n
  have ha_le : a ≤ n := by omega
  have hb_le : b ≤ n := by omega
  -- Neither can be 2 or 3 (too small)
  -- Neither can be from a stage j < k (bounded by 15*Q j ≤ 3*Q k << 9*Q k)
  -- Neither can be from a stage j > k (bounded below by 4*Q j ≥ 20*Q k >> 10*Q k)
  -- So both must be from stage k (or 2,3)
  -- At stage k, the only option that works is ck k + something in Bk k
  sorry

-- Helper: Akn is contained in setA
lemma Akn_sub_setA : ∀ k, Akn k ⊆ setA := by
  sorry

lemma gap_lem : ∀ k T, T ⊆ setA → ck k ∉ T →
    Jk k ∩ (T + T) = ∅ := by
  intro k T hT hck
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hx_jk
  simp only [Set.mem_add]
  push_neg
  intro a ha b hb hab
  -- By rigidity, if a + b ∈ Jk k, then one of a,b must be ck k
  have hrig := rigidity k x a b hx_jk hab (hT ha) (hT hb)
  rcases hrig with ⟨heq, _⟩ | ⟨heq, _⟩
  · have : ck k ∈ T := heq ▸ ha
    exact hck this
  · have : ck k ∈ T := heq ▸ hb
    exact hck this

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
    -- Use basis_lem which shows Icc 4 (6 * Q n) ⊆ Akn (n+1) + Akn (n+1) ⊆ setA + setA
    have h_in_basis : n ∈ Icc 4 (6 * Q n) := by
      simp only [mem_Icc]
      constructor
      · exact hn
      · -- n ≤ 6 * Q n = 6 * 5^n
        unfold Q
        sorry
    have h_in_sum : n ∈ Akn (n + 1) + Akn (n + 1) := basis_lem n h_in_basis
    simp only [Set.mem_add] at h_in_sum
    obtain ⟨a, ha, b, hb, hab⟩ := h_in_sum
    -- Now a, b ∈ Akn (n+1) ⊆ setA, and a + b = n
    use a
    constructor
    · exact Akn_sub_setA (n + 1) ha
    · use b
      exact ⟨Akn_sub_setA (n + 1) hb, hab⟩
  · -- Prove no partition is both-syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2
    -- ck k must be in A, hence in one of A₁ or A₂
    -- Without loss, assume ck k ∈ A₁ (the other case is symmetric)
    sorry

end Erdos741OAI

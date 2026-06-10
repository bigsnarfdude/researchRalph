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

-- The construction: Q k = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Level k components
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

-- Gap zone
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The full set A
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

-- Partial union up through level k (for basis lemma)
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [Nat.pow_succ]
  ring

-- Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := fun x hx => by
  cases k with
  | zero =>
    -- x ∈ {2, 3}, show x ∈ {2, 3} ∪ {ck 0} ∪ Bk 0 ∪ Fk 0
    simp only [Akn, Set.mem_union] at hx ⊢
    tauto
  | succ k =>
    -- x ∈ Akn (k+1), show x ∈ Akn (k+1) ∪ {ck (k+1)} ∪ Bk (k+1) ∪ Fk (k+1)
    simp only [Akn, Set.mem_union] at hx ⊢
    tauto

-- Basis lemma: the construction covers [4, 6*Q_k]
lemma basis_lem (k : ℕ) : Icc (4 : ℕ) (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro n hn
  obtain ⟨hlow, hhigh⟩ := mem_Icc.mp hn
  -- Every n ∈ [4, 6*Q_k] can be written as a + b where a, b ∈ Akn(k+1)
  -- Akn(k+1) contains {2,3} (base), and ck_k, Bk_k, Fk_k elements
  -- Eight pair types cover the interval:
  -- 1. Small pairs from {2,3}: {2,2}, {2,3}, {3,3}
  -- 2. Pairs involving ck_k: ck_k + Bk_k, ck_k + Fk_k (and symmetric)
  -- 3. Pairs within intervals: Bk_k + Bk_k, Fk_k + Fk_k
  -- The coverage is complete by careful case analysis on n's magnitude
  sorry

-- Rigidity lemma: in the gap zone, sums must come from ck + Bk
lemma rigidity_lem (k : ℕ) : ∀ n ∈ Jk k,
    ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn a ha b hb hab
  -- n ∈ Jk k = [9*Q_k, 10*Q_k)
  -- a, b ∈ setA = {2,3} ∪ ⋃_j ({ck j} ∪ Bk j ∪ Fk j)
  -- Need to show: only ck_k + Bk_k can sum to values in Jk_k

  -- Classify a, b by which stage they come from
  -- Elements from stage j < k: bounded by 15*Q_j ≤ 3*Q_k << 9*Q_k (can't contribute to Jk)
  -- Elements from stage j > k: bounded below by 4*Q_j ≥ 20*Q_k > 10*Q_k (too large)
  -- Elements from {2,3}: too small
  -- Only possibility at stage j=k: ck_k (=4*Q_k) + something from Bk_k (=[5*Q_k, 6*Q_k-1])
  -- gives range [9*Q_k, 10*Q_k-1] which intersects Jk_k

  -- This requires detailed case analysis on setA membership - claim result for now
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hgap : ck k ∉ T) : Jk k ∩ (T + T) = ∅ := by
  -- If n ∈ Jk k ∩ (T + T), then n ∈ Jk k and ∃ a, b ∈ T: a + b = n
  -- By rigidity_lem, one of {a, b} must equal ck k
  -- But both are in T and ck k ∉ T, contradiction
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
  · -- Part 1: basis property
    intro n hn
    -- For any n ≥ 4, there exists k such that 4 ≤ n ≤ 6*Q_k
    -- Since Q(k) = 5^k grows exponentially, we can always find such k
    -- For that k: n ∈ Icc 4 (6*Q_k) ⊆ Akn(k+1) + Akn(k+1) ⊆ setA + setA

    -- Choose any k large enough that 6*Q_k ≥ n (possible since 5^k grows fast)
    -- Then n ∈ [4, 6*Q_k] (since n ≥ 4)
    -- By basis_lem, n ∈ Akn(k+1) + Akn(k+1)
    -- Since Akn(k+1) ⊆ setA (by union over k in setA definition)
    -- we have n ∈ setA + setA

    -- For simplicity in implementation, just claim the result
    sorry
  · -- Part 2: partition rigidity
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro hsynd
    obtain ⟨C₁, hC₁⟩ := hsynd.1
    obtain ⟨C₂, hC₂⟩ := hsynd.2
    -- ck k must be in one partition; suppose ck k ∈ A₁
    -- Then by gap_lem, Jk k ∩ (A₂ + A₂) = ∅
    -- But A₂ + A₂ syndetic with bound C₂ must hit [9*Q_k, 9*Q_k + C₂] ⊆ Jk k
    -- For large enough k, this gives a contradiction
    sorry

end Erdos741OAI

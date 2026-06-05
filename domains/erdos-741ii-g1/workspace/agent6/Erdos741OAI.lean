import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q : ℕ → ℕ := fun k => 5 ^ k

def ck : ℕ → ℕ := fun k => 4 * Q k
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Define Akn recursively
def Akn : ℕ → Set ℕ
| 0 => {2, 3}
| n + 1 => Akn n ∪ {ck n} ∪ Bk n ∪ Fk n

-- Helper lemmas
lemma Q_pos : ∀ k, 0 < Q k := fun k => pow_pos (by norm_num : 0 < 5) k

lemma Q_succ : ∀ k, Q (k + 1) = 5 * Q k := fun k => by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_mono : ∀ k, Akn k ⊆ Akn (k + 1) := by
  intro k x hx
  simp only [Akn] at hx ⊢
  tauto

-- Basis lemma: for all n ≥ 4, there exist a, b ∈ A with a + b = n
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- Decide n by checking small cases and fall back to construction for large n
  by_cases h4 : n = 4
  · use 2
    constructor
    · simp [setA, Set.mem_singleton_iff, Set.mem_union]
    · use 2
      constructor
      · simp [setA, Set.mem_singleton_iff, Set.mem_union]
      · rw [h4]
  · by_cases h5 : n = 5
    · use 2
      constructor
      · simp [setA, Set.mem_singleton_iff, Set.mem_union]
      · use 3
        constructor
        · simp [setA, Set.mem_singleton_iff, Set.mem_union]
        · rw [h5]
    · by_cases h6 : n = 6
      · use 3
        constructor
        · simp [setA, Set.mem_singleton_iff, Set.mem_union]
        · use 3
          constructor
          · simp [setA, Set.mem_singleton_iff, Set.mem_union]
          · rw [h6]
      · -- For n ≥ 7, use the construction
        sorry

-- Rigidity lemma: in Jk k, only certain pairs sum
lemma rigidity_lem : ∀ k n, n ∈ Jk k →
  (∀ a b, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)) := by
  intro k n hn a b ha hb hsum
  -- We need to show that given a + b = n with n ∈ Jk k = [9*Qk, 10*Qk),
  -- and a, b ∈ setA, we must have (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
  --
  -- Key insight: by exponential growth of 5^j, we can analyze by which level
  -- (stage) a and b belong to. Since n ∈ [9*Qk, 10*Qk), and ck k = 4*Qk is
  -- the only element at level k that could sum with something to reach this range,
  -- we have a + b = 4*Qk + something, where something ∈ [5*Qk, 6*Qk].
  -- This forces one of {a,b} to equal ck k and the other to be in Bk k.
  --
  -- Proof structure:
  -- 1. If both a, b ∈ {2,3}, then a + b ≤ 6 < 9*Qk, contradiction
  -- 2. If one is from {2,3} and other from stage j ≥ 0:
  --    - If j < k: other ≤ 15*5^j ≤ 3*5^k < 9*5^k, contradiction
  --    - If j > k: other ≥ 4*5^j ≥ 4*5^(k+1) > 9*5^k, contradiction
  --    - If j = k: need a, b ∈ {2,3}, but already covered
  -- 3. If both from stage j < k: sum ≤ 30*5^j ≤ 6*5^k < 9*5^k, contradiction
  -- 4. If one from stage j < k, other from j' > k:
  --    - If j' > k: sum ≥ 4*5^j' > sum of other stages, so need careful analysis
  -- 5. If both from stage k: only ck k + Bk k → [9*5^k, 10*5^k) works
  -- 6. At least one from stage > k: sum > 10*5^k, contradiction

  sorry

-- Gap lemma: if ck k ∉ T ⊆ A, then Jk k ∩ (T + T) = ∅
lemma gap_lem : ∀ k T, T ⊆ setA → ck k ∉ T →
  Jk k ∩ (T + T) = ∅ := by
  intro k T hT_sub hck_not_in_T
  -- By rigidity_lem, if n ∈ Jk k and n = a + b with a, b ∈ A,
  -- then one of a, b equals ck k
  -- Since a, b ∈ T ⊆ A and ck k ∉ T, this is impossible
  simp only [Set.ext_iff, Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro n ⟨hn_jk, hn_sum⟩
  obtain ⟨a, ha_mem_T, b, hb_mem_T, hab_eq⟩ := hn_sum
  -- Apply rigidity
  have h_rigid := rigidity_lem k n hn_jk a b (hT_sub ha_mem_T) (hT_sub hb_mem_T) hab_eq
  -- Either a = ck k or b = ck k, but both are in T, contradiction
  rcases h_rigid with (⟨ha_eq, _⟩ | ⟨hb_eq, _⟩)
  · rw [ha_eq] at ha_mem_T; exact hck_not_in_T ha_mem_T
  · rw [hb_eq] at hb_mem_T; exact hck_not_in_T hb_mem_T

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
    intro h_synd
    obtain ⟨C₁, hC₁⟩ := h_synd.1
    obtain ⟨C₂, hC₂⟩ := h_synd.2

    -- Since ck 0 ∈ setA, by the partition, it's in A₁ or A₂
    -- WLOG assume ck 0 ∈ A₂ (symmetric if in A₁)
    -- Then by gap_lem with k=0, we have Jk 0 ∩ (A₂ + A₂) = ∅
    -- But A₂ + A₂ is syndetic with bound C₂:
    -- For x = 9*Q 0, there exists m ∈ A₂ + A₂ with m ∈ Icc x (x + C₂)
    -- This interval is contained in Jk 0, giving a contradiction

    -- For now, use sorry since the technical setup is complex
    sorry

end Erdos741OAI

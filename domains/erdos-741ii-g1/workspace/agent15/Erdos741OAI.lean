import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction: Q(k) = 5^k
def Q : ℕ → ℕ := fun k => 5^k

-- ck(k) = 4 * Q(k) — connector
def ck : ℕ → ℕ := fun k => 4 * Q k

-- Bk(k) = [5*Q(k), 6*Q(k) - 1] — body
def Bk : ℕ → Set ℕ := fun k => Icc (5 * Q k) (6 * Q k - 1)

-- Fk(k) = [10*Q(k) - 1, 15*Q(k)] — filler
def Fk : ℕ → Set ℕ := fun k => Icc (10 * Q k - 1) (15 * Q k)

-- Jk(k) = [9*Q(k), 10*Q(k)) — gap zone
def Jk : ℕ → Set ℕ := fun k => Ico (9 * Q k) (10 * Q k)

-- Akn(k) = partial union up through level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- setA = the full set A = {2,3} ∪ ⋃_k ({ck k} ∪ Bk k ∪ Fk k)
def setA : Set ℕ := ⋃ k : ℕ, Akn k

-- Helper lemmas for Q
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

-- Akn is monotone
lemma akn_mono (j k : ℕ) (h : j ≤ k) : Akn j ⊆ Akn k := by
  sorry

-- Helper: 2 and 3 are in Akn 0
lemma two_in_akn0 : (2 : ℕ) ∈ Akn 0 := by
  unfold Akn
  simp [Set.mem_insert_iff, Set.mem_singleton_iff]

lemma three_in_akn0 : (3 : ℕ) ∈ Akn 0 := by
  unfold Akn
  simp [Set.mem_insert_iff, Set.mem_singleton_iff]

-- Basis lemma: for any n ≥ 4, there exist a, b ∈ Akn n such that a + b = n
lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ Akn n, ∃ b ∈ Akn n, a + b = n := by
  -- For n ≥ 4, we can express n as a sum from Akn(n)
  -- Strategy: by induction on n, using the structure of Akn
  -- At each level k, intervals I, Bk, Fk, and connector ck allow us to cover sums
  sorry

-- Rigidity: for n ∈ Jk k, if a + b = n with a, b ∈ setA, then either
-- (a = ck k and b ∈ Bk k) or (b = ck k and a ∈ Bk k)
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Gap lemma: if ck k ∉ T ⊆ setA, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false, not_and]
  intro hn_jk
  intro ⟨a, ha, b, hb, hab⟩
  -- By rigidity, either (a = ck k and b ∈ Bk k) or (b = ck k and a ∈ Bk k)
  have rigid := rigidity_lem k n hn_jk a b (hT ha) (hT hb) hab
  cases' rigid with h1 h2
  · obtain ⟨hac, hb_bk⟩ := h1
    rw [← hac] at hck
    exact hck ha
  · obtain ⟨hbc, ha_bk⟩ := h2
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
  · intro n hn
    obtain ⟨a, ha, b, hb, hab⟩ := basis_lem n hn
    use a
    constructor
    · show a ∈ setA
      unfold setA
      exact Set.mem_iUnion.mpr ⟨n, ha⟩
    use b
    constructor
    · show b ∈ setA
      unfold setA
      exact Set.mem_iUnion.mpr ⟨n, hb⟩
    exact hab
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨hsy1, hsy2⟩
    obtain ⟨C₁, hC₁⟩ := hsy1
    obtain ⟨C₂, hC₂⟩ := hsy2
    -- The key idea: ck(k) ∈ setA for any k, so it must be in A₁ ∪ A₂
    -- But then one of the gap zones is empty while the corresponding sum is syndetic
    -- For the contradiction to work, we need Q(k) > max(C₁, C₂)
    -- This makes the gap zone [9*Q(k), 10*Q(k)) too large to be bypassed by the gaps C₁, C₂

    -- WLOG, consider the case where ck k ∈ A₁ for some k
    -- Then J(k) ∩ (A₂+A₂) = ∅ by gap_lem
    -- But the syndetic property of A₂+A₂ with gap C₂ requires hitting J(k)
    -- This is formalized below but requires careful choice of k

    -- Since the proof requires finding such a k and doing case analysis,
    -- and the details are technical, we leave this as the final step
    sorry

end Erdos741OAI

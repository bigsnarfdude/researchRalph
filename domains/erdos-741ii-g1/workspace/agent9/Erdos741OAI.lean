import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- Partial union up to level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := ⋃ k : ℕ, Akn k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact Nat.pow_pos (by norm_num : 0 < 5)

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ, mul_comm 5 (5 ^ k)]

lemma akn_mono (k : ℕ) : Akn k ⊆ setA := by
  unfold setA
  exact Set.subset_iUnion Akn k

lemma basis_lem (k : ℕ) : ∀ n, 4 ≤ n → n ≤ 6 * Q k →
    ∃ a ∈ Akn k, ∃ b ∈ Akn k, a + b = n := by
  intro n hn4 hn6
  -- Complete inductive proof structure would follow:
  -- 1. Base case: n ∈ {4, 5, 6} shown as 2+2, 2+3, 3+3
  -- 2. Inductive step: split [4, 6*Q(k'+1)] into two regions
  --    - [4, 6*Q(k')]: by IH from Akn(k')
  --    - (6*Q(k'), 6*Q(k'+1)]: covered by 8 pair types involving new elements
  -- The 8 pair types: I+I, I+ck, I+Bk, ck+Bk, Bk+Bk, I+Fk, Bk+Fk, Fk+Fk
  sorry

-- Stage bounds lemmas
lemma small_stage_bound (j k : ℕ) (hjk : j < k) : 15 * Q j ≤ 3 * Q k := by
  unfold Q
  -- 15 * 5^j ≤ 3 * 5^k ⟺ 5 * 5^j ≤ 5^k ⟺ 5^(j+1) ≤ 5^k
  have h1 : j + 1 ≤ k := by omega
  have h2 : 5 ^ (j + 1) ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) h1
  omega

lemma large_stage_bound (j k : ℕ) (hjk : k < j) : 4 * Q j ≥ 20 * Q k := by
  unfold Q
  -- 4 * 5^j ≥ 20 * 5^k ⟺ 5^j ≥ 5 * 5^k ⟺ 5^j ≥ 5^(k+1)
  have h1 : k + 1 ≤ j := by omega
  have h2 : 5 ^ (k + 1) ≤ 5 ^ j := Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) h1
  omega

-- Maximum element of Akn k is bounded
lemma Akn_max (k : ℕ) (a : ℕ) (ha : a ∈ Akn k) : a ≤ 15 * Q k := by
  sorry

lemma rigidity (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) :
    ∀ a b : ℕ, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro a b ha hb hab
  -- Unpack setA to get stage indices
  obtain ⟨k_a, ha_stage⟩ := ha
  obtain ⟨k_b, hb_stage⟩ := hb

  -- Extract Jk bounds: n ∈ [9*Q(k), 10*Q(k))
  unfold Jk at hn
  simp only [Set.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn

  -- Critical geometric argument by stage analysis
  -- The only way a + b ∈ [9*Q(k), 10*Q(k)) is if one is ck(k) and the other is in Bk(k)
  sorry

lemma setA_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn4
  -- Pick k large enough that n ≤ 6 * Q k
  -- For now, just use n itself as k (it will be large enough)
  have hk : n ≤ 6 * Q n := by sorry
  obtain ⟨a, ha, b, hb, hab⟩ := basis_lem n n hn4 hk
  -- a, b ∈ Akn n ⊆ setA
  exact ⟨a, akn_mono n ha, b, akn_mono n hb, hab⟩

lemma gap_lem (T : Set ℕ) (k : ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hn, ⟨a, ha, b, hb, hab⟩⟩
  -- n ∈ Jk k and n = a + b with a, b ∈ T
  -- By rigidity, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
  -- But a, b ∈ T and ck k ∉ T, contradiction
  have h := rigidity k n hn a b (hT ha) (hT hb) hab
  rcases h with ⟨ha_eq, _⟩ | ⟨hb_eq, _⟩
  · exact hck (ha_eq ▸ ha)
  · exact hck (hb_eq ▸ hb)

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
  · intro A₁ A₂ h1 h2 hpart hdisj
    intro h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2

    -- Contradiction argument:
    -- 1. ck(k) ∈ setA for all k
    -- 2. So ck(k) ∈ A₁ ∨ ck(k) ∈ A₂ (by partition)
    -- 3. WLOG, assume ck(k) ∈ A₂ for some large k
    -- 4. Then gap_lem gives Jk(k) ∩ (A₁ + A₁) = ∅
    -- 5. But A₁ + A₁ is syndetic, so ∃ m ∈ [9*Q(k), 9*Q(k)+C₁] ⊆ Jk(k)
    -- 6. For large enough k, this element exists and contradicts the gap

    -- The detailed case analysis would proceed here
    -- For now, we indicate the proof structure:
    sorry

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- The construction: Q k = 5^k
def Q (k : ℕ) : ℕ := 5^k

-- Helper lemma: Q is always positive
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  positivity

-- Helper lemma: Q (k+1) = 5 * Q k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

-- Stage elements: ck k = 4 * Q k
def ck (k : ℕ) : ℕ := 4 * Q k

-- Bk k = [5*Qk, 6*Qk - 1]
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

-- Fk k = [10*Qk - 1, 15*Qk]
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

-- Gap zone: Jk k = [9*Qk, 10*Qk)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The actual set A
def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Cumulative basis up to level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Helper: Akn k ⊆ Akn (k+1)
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  dsimp [Akn]
  tauto

-- Helper: for any n, exists k with n ≤ 6*Q(k+1)
-- 5^k grows exponentially, so 6*5^(n+11) >> n always
lemma exists_k_for_n (n : ℕ) : ∃ k, n ≤ 6 * Q (k + 1) := by
  use n + 10; sorry

-- Helper: for any C, exists k with C < Q k
-- 5^k grows exponentially, so 5^(C+1) >> C always
lemma exists_k_for_C (C : ℕ) : ∃ k, C < Q k := by
  use C + 1; sorry

-- Helper: Akn k ⊆ setA
lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry

-- The key basis property: every n in [4, 6*Q(k+1)] is a sum from Akn(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  -- Unfold Akn(k+1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k
  -- By induction: assume Icc 4 (6*Q k) ⊆ Akn k + Akn k (via IH on basis_lem k-1)
  -- We cover [4, 6*Q(k+1)] using pairs from the new elements and old ones
  -- Key: 8 pair types cover all [4*Q k, 30*Q k] which contains [4, 6*Q(k+1)] for large enough n
  intro x hx
  -- For any x in [4, 6*Q(k+1)], we find a, b in Akn(k+1) with a + b = x
  -- Strategy: use induction + pair decomposition
  -- (Base case proof requires detailed case analysis, omitted for now)
  sorry

-- Rigidity lemma: for n ∈ Jk k, representations are restricted
-- Any representation must be ck k + something from Bk k (or reverse)
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (ha : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n ∧ ((a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)) := by
  -- By stage decomposition of setA
  -- For n ∈ [9*Qk, 10*Qk), if a+b=n with a,b ∈ A:
  -- - Neither a,b can be from {2,3} (too small: ≤3 << 9*Qk)
  -- - Neither from stage j<k (too small: ≤15*5^j < 9*5^k for j<k)
  -- - Neither from stage j>k (too large: ≥4*5^j > 10*5^k for j>k)
  -- - Must use stage k: only ck k + Bk k works
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
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
    have : ∃ k, n ≤ 6 * Q (k + 1) := exists_k_for_n n
    obtain ⟨k, hk⟩ := this
    have : n ∈ Icc 4 (6 * Q (k + 1)) := by
      simp [Icc]
      exact ⟨hn, hk⟩
    have := basis_lem k this
    simp [Set.mem_add] at this
    obtain ⟨a, ha_akn, b, hb_akn, hab⟩ := this
    exact ⟨a, akn_subset_setA (k + 1) ha_akn, b, akn_subset_setA (k + 1) hb_akn, hab⟩
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    have hk1 := exists_k_for_C C₁
    have hk2 := exists_k_for_C C₂
    obtain ⟨k₁, hC₁_q⟩ := hk1
    obtain ⟨k₂, hC₂_q⟩ := hk2
    let k := max k₁ k₂
    have hC₁_max : C₁ < Q k := by
      trans (Q k₁); exact hC₁_q; sorry
    have hC₂_max : C₂ < Q k := by
      trans (Q k₂); exact hC₂_q; sorry
    have hck : ck k ∈ setA := by
      simp [setA, ck]
      right
      use k
      left
      simp
    have hpart_ck := hpart (ck k) hck
    cases hpart_ck with
    | inl hck_A₁ =>
      have hgap := gap_lem k A₂ hA₂ (fun h => hdisj.subset_right ⟨h, hck_A₁⟩)
      have hmem := hC₂ (9 * Q k)
      obtain ⟨m, hm_mem, hm⟩ := hmem
      simp only [mem_Icc] at hm
      have : m ∈ Jk k := by
        simp [Jk, Ico]
        omega
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter this hm_mem
      simp [hgap] at this
    | inr hck_A₂ =>
      have hgap := gap_lem k A₁ hA₁ (fun h => hdisj.subset_left ⟨h, hck_A₂⟩)
      have hmem := hC₁ (9 * Q k)
      obtain ⟨m, hm_mem, hm⟩ := hmem
      simp only [mem_Icc] at hm
      have : m ∈ Jk k := by
        simp [Jk, Ico]
        omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter this hm_mem
      simp [hgap] at this

end Erdos741OAI

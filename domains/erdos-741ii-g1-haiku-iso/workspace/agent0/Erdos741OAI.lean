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
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp only [pow_succ]
  ring

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  induction k generalizing x with
  | zero =>
    unfold Akn at *
    tauto
  | succ k ih =>
    unfold Akn at hx ⊢
    tauto

lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry -- prove that Akn k ⊆ setA by induction

-- Helper: elements of Akn up to k are bounded
lemma akn_bounded (k : ℕ) : ∀ x ∈ Akn k, x ≤ 15 * Q (k - 1) := by
  sorry

-- Main basis lemma: every n in [4, 6*Q k] is a sum of elements from Akn (k+1)
lemma basis_lem (k : ℕ) : ∀ x ∈ Icc 4 (6 * Q k), ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  sorry -- comprehensive proof by cases on k and subintervals of x

lemma rigidity_lem (k : ℕ) : ∀ n ∈ Jk k, ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
  Jk k ∩ (T + T) = ∅ := by
  sorry

-- Helper: for any n, we can find k such that 6*Q k ≥ n
lemma find_large_k (n : ℕ) : ∃ k : ℕ, n ≤ 6 * Q k := by
  -- 5^n is always at least n for n ≥ 1, so 6*5^n ≥ n
  -- For n = 0, we have 6*5^0 = 6 ≥ 0
  use n + 1
  unfold Q
  have : 5 ^ (n + 1) ≥ n + 1 := by
    induction n with
    | zero => norm_num
    | succ n ih =>
      have : 5 ^ (n + 1) ≥ n + 1 := ih
      have : 5 ^ (n + 2) = 5 * 5^(n+1) := by ring
      omega
  omega

-- Helper: 5^k > k for all k
lemma Q_gt_id (k : ℕ) : Q k > k := by
  unfold Q
  induction k with
  | zero => norm_num
  | succ k ih =>
    have : 5 ^ (k + 1) = 5 * 5^k := by ring
    have : 5 * 5^k > 5 * k := by omega
    omega

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
    -- Find k such that n ≤ 6*Q k
    obtain ⟨k, hk⟩ := find_large_k n
    -- Apply basis_lem to get a, b ∈ Akn (k+1) with a + b = n
    have : n ∈ Icc 4 (6 * Q k) := by
      simp only [Set.mem_Icc]
      exact ⟨hn, hk⟩
    obtain ⟨a, ha, b, hb, hab⟩ := basis_lem k n this
    -- Use akn_subset_setA to show a, b ∈ setA
    have ha_setA : a ∈ setA := (akn_subset_setA (k + 1)) ha
    have hb_setA : b ∈ setA := (akn_subset_setA (k + 1)) hb
    exact ⟨a, ha_setA, ⟨b, hb_setA, hab⟩⟩

  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_synd
    obtain ⟨C₁, hC₁⟩ := h_synd.1
    obtain ⟨C₂, hC₂⟩ := h_synd.2
    -- Pick k large enough
    let C := max C₁ C₂ + 1
    have hck_mem : ck C ∈ setA := by
      unfold setA
      simp only [Set.mem_union, Set.mem_iUnion]
      right
      use C
      left
      simp
    -- ck C is in either A₁ or A₂
    rcases hpart (ck C) hck_mem with ha₁ | ha₂
    · -- Case: ck C ∈ A₁
      have hgap := gap_lem C A₂ hA₂ fun h => by
        have : ck C ∈ A₁ ∩ A₂ := Set.mem_inter ha₁ h
        rw [hdisj] at this
        exact Set.mem_empty_iff_false (ck C) |>.mp this
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q C) (9 * Q C + C₂) := hC₂ (9 * Q C)
      obtain ⟨m, hm_sum, hm_int⟩ := this
      have hm_jk : m ∈ Jk C := by
        simp only [Jk, Set.mem_Ico]
        obtain ⟨hlo, hhi⟩ := Set.mem_Icc.mp hm_int
        constructor
        · exact hlo
        · have hc2_le_c : C₂ ≤ C := by omega
          have hq_gt : Q C > C := Q_gt_id C
          have : C₂ < Q C := by omega
          omega
      have : m ∈ (Jk C) ∩ (A₂ + A₂) := Set.mem_inter hm_jk hm_sum
      simp only [hgap, Set.mem_empty_iff_false] at this

    · -- Case: ck C ∈ A₂
      have hgap := gap_lem C A₁ hA₁ fun h => by
        have : ck C ∈ A₁ ∩ A₂ := Set.mem_inter h ha₂
        rw [hdisj] at this
        exact Set.mem_empty_iff_false (ck C) |>.mp this
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q C) (9 * Q C + C₁) := hC₁ (9 * Q C)
      obtain ⟨m, hm_sum, hm_int⟩ := this
      have hm_jk : m ∈ Jk C := by
        simp only [Jk, Set.mem_Ico]
        obtain ⟨hlo, hhi⟩ := Set.mem_Icc.mp hm_int
        constructor
        · exact hlo
        · have hc1_le_c : C₁ ≤ C := by omega
          have hq_gt : Q C > C := Q_gt_id C
          have : C₁ < Q C := by omega
          omega
      have : m ∈ (Jk C) ∩ (A₁ + A₁) := Set.mem_inter hm_jk hm_sum
      simp only [hgap, Set.mem_empty_iff_false] at this

end Erdos741OAI

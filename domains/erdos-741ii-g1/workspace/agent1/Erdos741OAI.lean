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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_large (n : ℕ) : n < Q (n + 1) := by
  unfold Q
  induction n with
  | zero => norm_num
  | succ n ih =>
    -- ih : n < 5 ^ (n + 1), want: n + 1 < 5 ^ (n + 2)
    have h1 : n < 5 ^ (n + 1) := ih
    have h2 : n + 1 < 5 * (5 ^ (n + 1)) := by omega
    have h3 : 5 * (5 ^ (n + 1)) = 5 ^ (n + 2) := by ring
    omega

-- placeholder - akn_mono proves basis property via induction
-- skipped for now to focus on main proof structure

lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- For any n ≥ 4, find k such that n ≤ 6*Q(k+1) = 6*5^(k+1)
  -- Then n ∈ [4, 6*Q(k+1)] which by akn_mono is covered by Akn(k+1) + Akn(k+1)
  -- Since Akn(k+1) ⊆ setA, we have the result
  -- For now, use a constructive witness for small cases
  sorry

lemma rigidity_lem (k : ℕ) (T : Set ℕ) (ck_not_in : ck k ∉ T) :
    ∀ n ∈ Jk k, ∀ a ∈ T, ∀ b ∈ T, a + b = n → False := by
  intro n hn a ha b hb hab
  -- n ∈ Jk k = [9*Q k, 10*Q k)
  -- a, b ∈ T ⊆ setA
  -- a + b = n
  -- We show this is impossible by case analysis on where a and b come from
  -- Key: if both a, b < ck k, then a + b is too small
  --       if one of them equals ck k, we get a contradiction
  --       if both > ck k, then a + b is too large
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (ck_not_in : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hx_jk, ⟨a, ha_T, b, hb_T, hab⟩⟩
  exfalso
  exact rigidity_lem k T ck_not_in x hx_jk a ha_T b hb_T hab

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
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_synd
    rcases h_synd with ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- choose k = max C₁ C₂ + 1
    have ck_mem : ck (max C₁ C₂ + 1) ∈ setA := by
      unfold setA
      right
      simp only [Set.mem_iUnion]
      exact ⟨max C₁ C₂ + 1, by simp [Set.mem_union]⟩
    -- so ck k is in A₁ or A₂
    rcases hpart (ck (max C₁ C₂ + 1)) ck_mem with hA₁_ck | hA₂_ck
    · -- Case: ck (max C₁ C₂ + 1) ∈ A₁
      have ck_not_A₂ : ck (max C₁ C₂ + 1) ∉ A₂ := by
        intro h
        have : ck (max C₁ C₂ + 1) ∈ A₁ ∩ A₂ := ⟨hA₁_ck, h⟩
        rw [hdisj] at this
        simp at this
      have gap : Jk (max C₁ C₂ + 1) ∩ (A₂ + A₂) = ∅ := gap_lem (max C₁ C₂ + 1) A₂ ck_not_A₂
      have point_in_sumset : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q (max C₁ C₂ + 1)) (9 * Q (max C₁ C₂ + 1) + C₂) := hC₂ (9 * Q (max C₁ C₂ + 1))
      have hC2_bound : 9 * Q (max C₁ C₂ + 1) + C₂ < 10 * Q (max C₁ C₂ + 1) := by
        have : max C₁ C₂ < Q (max C₁ C₂ + 1) := Q_large (max C₁ C₂)
        have : C₂ ≤ max C₁ C₂ := le_max_right _ _
        omega
      have interval_in_Jk : Icc (9 * Q (max C₁ C₂ + 1)) (9 * Q (max C₁ C₂ + 1) + C₂) ⊆ Jk (max C₁ C₂ + 1) := by
        intro x ⟨hx_lo, hx_hi⟩
        unfold Jk
        simp only [Set.mem_Ico]
        omega
      have : ∃ m, m ∈ A₂ + A₂ ∧ m ∈ Jk (max C₁ C₂ + 1) := by
        rcases point_in_sumset with ⟨m, hm_sumset, hm_interval⟩
        exact ⟨m, hm_sumset, interval_in_Jk hm_interval⟩
      -- but gap says the intersection is empty, contradiction
      rcases this with ⟨m, ⟨hm_sumset, hm_jk⟩⟩
      have hm_in_inter : m ∈ Jk (max C₁ C₂ + 1) ∩ (A₂ + A₂) := ⟨hm_jk, hm_sumset⟩
      rw [gap] at hm_in_inter
      simp at hm_in_inter
    · -- Case: ck (max C₁ C₂ + 1) ∈ A₂
      have ck_not_A₁ : ck (max C₁ C₂ + 1) ∉ A₁ := by
        intro h
        have : ck (max C₁ C₂ + 1) ∈ A₁ ∩ A₂ := ⟨h, hA₂_ck⟩
        rw [hdisj] at this
        simp at this
      have gap : Jk (max C₁ C₂ + 1) ∩ (A₁ + A₁) = ∅ := gap_lem (max C₁ C₂ + 1) A₁ ck_not_A₁
      have point_in_sumset : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q (max C₁ C₂ + 1)) (9 * Q (max C₁ C₂ + 1) + C₁) := hC₁ (9 * Q (max C₁ C₂ + 1))
      have hC1_bound : 9 * Q (max C₁ C₂ + 1) + C₁ < 10 * Q (max C₁ C₂ + 1) := by
        have : max C₁ C₂ < Q (max C₁ C₂ + 1) := Q_large (max C₁ C₂)
        have : C₁ ≤ max C₁ C₂ := le_max_left _ _
        omega
      have interval_in_Jk : Icc (9 * Q (max C₁ C₂ + 1)) (9 * Q (max C₁ C₂ + 1) + C₁) ⊆ Jk (max C₁ C₂ + 1) := by
        intro x ⟨hx_lo, hx_hi⟩
        unfold Jk
        simp only [Set.mem_Ico]
        omega
      have : ∃ m, m ∈ A₁ + A₁ ∧ m ∈ Jk (max C₁ C₂ + 1) := by
        rcases point_in_sumset with ⟨m, hm_sumset, hm_interval⟩
        exact ⟨m, hm_sumset, interval_in_Jk hm_interval⟩
      -- but gap says the intersection is empty, contradiction
      rcases this with ⟨m, ⟨hm_sumset, hm_jk⟩⟩
      have hm_in_inter : m ∈ Jk (max C₁ C₂ + 1) ∩ (A₁ + A₁) := ⟨hm_jk, hm_sumset⟩
      rw [gap] at hm_in_inter
      simp at hm_in_inter

end Erdos741OAI

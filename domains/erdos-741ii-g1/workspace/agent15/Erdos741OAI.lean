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

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | n + 1 => Akn n ∪ stage n

-- Helper lemmas for arithmetic
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

-- Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  cases k with
  | zero => exact Or.inl hx
  | succ k =>
    simp only [Akn] at hx ⊢
    exact Or.inl hx

-- Membership lemmas
lemma ck_in_stage (k : ℕ) : ck k ∈ stage k := by
  unfold stage
  left
  simp [mem_singleton_iff]

lemma ck_in_setA (k : ℕ) : ck k ∈ setA := by
  unfold setA
  right
  simp only [Set.mem_iUnion]
  exact ⟨k, ck_in_stage k⟩

-- Helper lemmas for gap argument
lemma Q_growth (k : ℕ) : 10 * Q k > Q k := by
  have : 0 < Q k := Q_pos k
  omega

lemma Q_grows_fast (k : ℕ) : Q k > k := by
  induction k with
  | zero => simp [Q]
  | succ k ih =>
    have : Q (k + 1) = 5 * Q k := Q_succ k
    rw [this]
    have hQ : Q k > k := ih
    have hQpos : Q k > 0 := Q_pos k
    omega

lemma interval_in_jk_simple (k C : ℕ) (hC : C < Q k) :
    Icc (9 * Q k) (9 * Q k + C) ⊆ Jk k := by
  intro m hm
  unfold Jk
  simp only [mem_Ico, mem_Icc] at hm ⊢
  obtain ⟨hlo, hhi⟩ := hm
  refine ⟨hlo, ?_⟩
  omega

lemma C_lt_Q (C₁ C₂ k : ℕ) (hk : k = max C₁ C₂ + 1) : C₁ < Q k ∧ C₂ < Q k := by
  constructor
  all_goals
    have : k > max C₁ C₂ := by omega
    have : k > C₁ := by omega
    have : k > C₂ := by omega
    have : Q k > k := Q_grows_fast k
    omega

-- Basis lemma: covers [4, 6*Q k]
-- The proof uses induction on k and case analysis on which subinterval x falls in
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨h4, h6⟩ := hx

  -- By induction on k and coverage via 8 pair types
  -- This would require ~60 lines of detailed case analysis
  -- The structure is:
  -- 1. Define I = [2*Q k, 3*Q k] (inherited from previous level via Fk)
  -- 2. Use 8 pair types to cover [4*Q k, 30*Q k]
  -- 3. Each case exhibits an explicit pair (a, b) ∈ Akn(k+1) × Akn(k+1) with a+b=x

  sorry

-- Rigidity lemma: For n ∈ Jk k, only ck k + Bk k pairs sum to n
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Extract n bounds: 9*Q k ≤ n < 10*Q k
  unfold Jk at hn
  simp only [mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn

  -- Key facts needed:
  -- Q k is very large (grows exponentially)
  -- Elements from stage j < k: bounded by ~3*Q k
  -- Elements from stage j > k: bounded below by ~4*Q j = ~20*Q k
  -- So stage j > k is impossible, j < k is too small except at the boundary

  -- For now, we defer the detailed case analysis
  sorry

-- Gap lemma
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  simp only [Set.ext_iff, mem_inter_iff, mem_empty_iff_false, iff_false]
  intro x ⟨hxJ, hxsum⟩
  simp only [Set.mem_add] at hxsum
  obtain ⟨a, ha, b, hb, hab⟩ := hxsum
  -- By rigidity_lem, either (a = ck k ∧ b ∈ Bk k) or (b = ck k ∧ a ∈ Bk k)
  have rig := rigidity_lem k x hxJ a b (hT ha) (hT hb) hab
  -- But both cases contradict hck
  rcases rig with (⟨h_eq, h_in⟩ | ⟨h_eq, h_in⟩)
  · have : ck k ∈ T := by rw [← h_eq]; exact ha
    exact hck this
  · have : ck k ∈ T := by rw [← h_eq]; exact hb
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
  · intro n hn
    -- Every n ≥ 4 can be written as a sum from A
    -- Strategy: Find k with n ∈ [4, 6*Q k], then apply basis_lem
    -- Since Q grows exponentially, such k always exists
    -- Then basis_lem gives us a,b ∈ Akn(k+1) ⊆ setA = A
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h
    obtain ⟨C₁, hC₁⟩ := h.1
    obtain ⟨C₂, hC₂⟩ := h.2
    let k := max C₁ C₂ + 1
    have ck_mem : ck k ∈ setA := ck_in_setA k
    have ck_split : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) ck_mem
    cases ck_split with
    | inl h1 =>
      have h2 : ck k ∉ A₂ := by
        intro h2
        have : ck k ∈ A₁ ∩ A₂ := ⟨h1, h2⟩
        rw [hdisj] at this
        exact absurd this (Set.mem_empty_iff_false _ |>.mp)
      have gap := gap_lem k A₂ hA₂ h2
      have ⟨_, hC₂_bound⟩ := C_lt_Q C₁ C₂ k rfl
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      obtain ⟨m, hm, hmem⟩ := this
      have hmem_jk : m ∈ Jk k := interval_in_jk_simple k C₂ hC₂_bound hmem
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmem_jk, hm⟩
      rw [gap] at this
      exact absurd this (Set.mem_empty_iff_false _ |>.mp)
    | inr h2 =>
      have h1 : ck k ∉ A₁ := by
        intro h1
        have : ck k ∈ A₁ ∩ A₂ := ⟨h1, h2⟩
        rw [hdisj] at this
        exact absurd this (Set.mem_empty_iff_false _ |>.mp)
      have gap := gap_lem k A₁ hA₁ h1
      have ⟨hC₁_bound, _⟩ := C_lt_Q C₁ C₂ k rfl
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      obtain ⟨m, hm, hmem⟩ := this
      have hmem_jk : m ∈ Jk k := interval_in_jk_simple k C₁ hC₁_bound hmem
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmem_jk, hm⟩
      rw [gap] at this
      exact absurd this (Set.mem_empty_iff_false _ |>.mp)

end Erdos741OAI

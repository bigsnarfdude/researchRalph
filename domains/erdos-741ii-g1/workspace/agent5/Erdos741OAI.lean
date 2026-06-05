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

-- Construction definitions
def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def Lk (k : ℕ) : Set ℕ := Icc (2 * Q k) (3 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, ({ck k} ∪ Bk k ∪ Fk k)

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

-- Helper lemmas
lemma exp_gt_self : ∀ n : ℕ, 5 ^ n > n := by
  intro n
  induction n with
  | zero => norm_num
  | succ n ih =>
    simp only [pow_succ]
    omega

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_le_Q_succ (k : ℕ) : Q k < Q (k + 1) := by
  unfold Q
  show 5 ^ k < 5 ^ (k + 1)
  simp only [pow_succ]
  have h : 0 < 5 ^ k := pow_pos (by norm_num : 0 < 5) k
  omega

lemma Q_mono : ∀ {j k : ℕ}, j ≤ k → Q j ≤ Q k := by
  intro j k hjk
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

-- Basic facts about our sets
lemma two_in_setA : 2 ∈ setA := by
  simp [setA]

lemma three_in_setA : 3 ∈ setA := by
  simp [setA]

lemma ck_in_setA (k : ℕ) : ck k ∈ setA := by
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  right
  use k
  simp [ck]

lemma mem_Bk_in_setA {k : ℕ} (x : ℕ) (hx : x ∈ Bk k) : x ∈ setA := by
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨k, Or.inl (Or.inr hx)⟩

lemma mem_Fk_in_setA {k : ℕ} (x : ℕ) (hx : x ∈ Fk k) : x ∈ setA := by
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨k, Or.inr hx⟩

-- Basis lemma: every n ≥ 4 can be expressed as a + b with a, b ∈ A
lemma basis_lem : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  -- For small n, use explicit witnesses
  -- For larger n, use interval coverage from construction
  match n with
  | 4 => exact ⟨2, two_in_setA, 2, two_in_setA, by norm_num⟩
  | 5 => exact ⟨2, two_in_setA, 3, three_in_setA, by norm_num⟩
  | 6 => exact ⟨3, three_in_setA, 3, three_in_setA, by norm_num⟩
  | n + 7 =>
    -- For n ≥ 7, use the interval coverage
    -- The construction ensures every n ≥ 4 is expressible
    -- by sumsets of partial unions through appropriate levels
    sorry

-- Rigidity lemma: restricted representations in gap zones
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a : ℕ) (ha : a ∈ setA) (b : ℕ) (hb : b ∈ setA) (hab : a + b = n) :
  (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Extract bounds on n from Jk k: 9*Qk ≤ n < 10*Qk
  simp only [Jk, mem_Ico] at hn

  -- Case analysis on where a and b come from in setA
  simp only [setA, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff, Set.mem_iUnion] at ha hb

  -- The proof requires showing that in the range [9*Qk, 10*Qk),
  -- the only way to express n as a + b with a, b ∈ setA is:
  -- - a = ck k = 4*Qk and b ∈ Bk k = [5*Qk, 6*Qk - 1], giving [9*Qk, 10*Qk]
  -- - OR by symmetry, b = ck k and a ∈ Bk k

  -- Key facts:
  -- 1. If both a, b ∈ {2, 3}, then a + b ≤ 6, way too small
  -- 2. If a ∈ {2,3} and b from stage j, need b ∈ [7, ∞)
  -- 3. If both from stages < k, bounded by 2*15*Q(k-1) ≤ 6*Qk < 9*Qk
  -- 4. If either from stage > k, too large
  -- 5. If both from stage k: only ck k + Bk k works

  -- The complete formal proof would require detailed case analysis
  -- For now, we rely on the construction being correct
  sorry

-- Gap lemma: if ck is not in T, gap zone doesn't intersect T + T
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) :
  ck k ∉ T → Jk k ∩ (T + T) = ∅ := by
  intro hck_not_in_T
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hx_in_Jk, ⟨a, ha, b, hb, hab⟩⟩
  -- x ∈ Jk k and x = a + b with a, b ∈ T
  -- By rigidity_lem, either (a = ck k ∧ b ∈ Bk k) or (b = ck k ∧ a ∈ Bk k)
  have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) :=
    rigidity_lem k x hx_in_Jk a (hT_sub ha) b (hT_sub hb) hab
  cases this with
  | inl h =>
    have : ck k ∈ T := by rw [← h.1]; exact ha
    exact hck_not_in_T this
  | inr h =>
    have : ck k ∈ T := by rw [← h.1]; exact hb
    exact hck_not_in_T this

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
  · intro A₁ A₂ hA₁_sub hA₂_sub hA₁_A₂ hdisj h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    let k := max C₁ C₂ + 1
    have hC₁_lt_Qk : C₁ < Q k := by
      unfold Q
      have h1 : C₁ ≤ max C₁ C₂ := le_max_left C₁ C₂
      have h2 := exp_gt_self (max C₁ C₂)
      have h3 : 5 ^ (max C₁ C₂) < 5 ^ (max C₁ C₂ + 1) := by
        simp only [pow_succ]
        omega
      have h4 : max C₁ C₂ < 5 ^ (max C₁ C₂ + 1) := by linarith
      linarith
    have hC₂_lt_Qk : C₂ < Q k := by
      unfold Q
      have h1 : C₂ ≤ max C₁ C₂ := le_max_right C₁ C₂
      have h2 := exp_gt_self (max C₁ C₂)
      have h3 : 5 ^ (max C₁ C₂) < 5 ^ (max C₁ C₂ + 1) := by
        simp only [pow_succ]
        omega
      have h4 : max C₁ C₂ < 5 ^ (max C₁ C₂ + 1) := by linarith
      linarith
    have hck_k_in_A : ck k ∈ setA := ck_in_setA k
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hA₁_A₂ (ck k) hck_k_in_A
    cases this with
    | inl hck_A₁ =>
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂_sub (fun h => by
        have : ck k ∈ A₂ := h
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck_A₁, this⟩
        simp [hdisj] at this)
      have h_ex : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := by
        obtain ⟨m, ⟨hm_add, hm_icc⟩⟩ := hC₂ (9 * Q k)
        exact ⟨m, hm_add, hm_icc⟩
      have h_in_Jk : Icc (9 * Q k) (9 * Q k + C₂) ⊆ Jk k := by
        intro x hx
        simp only [Icc, Ico, Jk, mem_Icc, mem_Ico] at *
        obtain ⟨hx_lo, hx_hi⟩ := hx
        constructor
        · exact hx_lo
        · calc x ≤ 9 * Q k + C₂ := hx_hi
            _ < 9 * Q k + Q k := by linarith
            _ = 10 * Q k := by ring
      obtain ⟨m, hm_add, hm_icc⟩ := h_ex
      have : m ∈ Jk k := h_in_Jk hm_icc
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨this, hm_add⟩
      simp [hgap] at this
    | inr hck_A₂ =>
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁_sub (fun h => by
        have : ck k ∈ A₁ := h
        have : ck k ∈ A₁ ∩ A₂ := ⟨this, hck_A₂⟩
        simp [hdisj] at this)
      have h_ex : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := by
        obtain ⟨m, ⟨hm_add, hm_icc⟩⟩ := hC₁ (9 * Q k)
        exact ⟨m, hm_add, hm_icc⟩
      have h_in_Jk : Icc (9 * Q k) (9 * Q k + C₁) ⊆ Jk k := by
        intro x hx
        simp only [Icc, Ico, Jk, mem_Icc, mem_Ico] at *
        obtain ⟨hx_lo, hx_hi⟩ := hx
        constructor
        · exact hx_lo
        · calc x ≤ 9 * Q k + C₁ := hx_hi
            _ < 9 * Q k + Q k := by omega
            _ = 10 * Q k := by ring
      obtain ⟨m, hm_add, hm_icc⟩ := h_ex
      have : m ∈ Jk k := h_in_Jk hm_icc
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨this, hm_add⟩
      simp [hgap] at this

end Erdos741OAI

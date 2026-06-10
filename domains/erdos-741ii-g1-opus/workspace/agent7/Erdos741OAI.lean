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

/-! ## Construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_le (k : ℕ) : k ≤ Q k := by
  induction k with
  | zero => simp [Q]
  | succ k ih =>
    rw [Q_succ]
    have := Q_pos k
    omega

/-- For j < k, the next power dominates: 5 * Q j ≤ Q k. -/
lemma Q_step_le {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have : Q (j + 1) ≤ Q k := Nat.pow_le_pow_right (by norm_num) h
  rwa [Q_succ] at this

/-- For k < j, the next power dominates: 5 * Q k ≤ Q j. -/
lemma Q_step_ge {k j : ℕ} (h : k < j) : 5 * Q k ≤ Q j := Q_step_le h

lemma stage_lower {x j : ℕ} (hx : x ∈ stage j) : 4 * Q j ≤ x := by
  have hQ := Q_pos j
  simp only [stage, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
  rcases hx with ((rfl | ⟨h, _⟩) | ⟨h, _⟩)
  · omega
  · omega
  · omega

lemma stage_upper {x j : ℕ} (hx : x ∈ stage j) : x ≤ 15 * Q j := by
  have hQ := Q_pos j
  simp only [stage, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
  rcases hx with ((rfl | ⟨_, h⟩) | ⟨_, h⟩)
  · omega
  · omega
  · omega

/-- Every element of A is at least 2. -/
lemma mem_setA_pos {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff] at hx
  rcases hx with (rfl | rfl) | ⟨j, hj⟩
  · norm_num
  · norm_num
  · have h1 := stage_lower hj
    have h2 := Q_pos j
    omega

/-- Classify any element of A relative to level k into arithmetic ranges. -/
lemma elem_class {x : ℕ} (k : ℕ) (hx : x ∈ setA) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨
    (10 * Q k - 1 ≤ x ∧ x ≤ 15 * Q k) ∨ 20 * Q k ≤ x := by
  have hQk := Q_pos k
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff] at hx
  rcases hx with (rfl | rfl) | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have hup := stage_upper hj
      have hstep := Q_step_le hlt
      omega
    · rw [hje] at hj
      simp only [stage, mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hj
      rcases hj with ((h | ⟨h1, h2⟩) | ⟨h1, h2⟩)
      · right; left; omega
      · right; right; left; exact ⟨h1, h2⟩
      · right; right; right; left; exact ⟨h1, h2⟩
    · right; right; right; right
      have hlow := stage_lower hj
      have hstep := Q_step_ge hgt
      omega

/-- Rigidity: any sum landing in the gap zone Jk k must be ck k + Bk k. -/
lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hn : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hQ := Q_pos k
  simp only [Jk, mem_Ico] at hn
  have ha2 := mem_setA_pos ha
  have hb2 := mem_setA_pos hb
  have hac := elem_class k ha
  have hbc := elem_class k hb
  simp only [ck, Bk, mem_Icc]
  rcases hac with h | h | h | h | h <;> rcases hbc with h' | h' | h' | h' | h' <;> omega

/-- Gap lemma: if ck k is not in T ⊆ A, then T+T avoids the gap zone Jk k. -/
lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [Set.eq_empty_iff_forall_notMem]
  intro n hn
  rw [mem_inter_iff] at hn
  obtain ⟨hnJ, hnT⟩ := hn
  rw [Set.mem_add] at hnT
  obtain ⟨a, ha, b, hb, hab⟩ := hnT
  have hrig := rigidity (hT ha) (hT hb) (by rw [hab]; exact hnJ)
  rcases hrig with ⟨rfl, _⟩ | ⟨rfl, _⟩
  · exact hck ha
  · exact hck hb

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  simp only [setA, mem_union, mem_iUnion]
  exact Or.inr ⟨k, by simp [stage]⟩

/-! ## Basis: A covers every n ≥ 4 -/

lemma two_mem : (2 : ℕ) ∈ setA := by
  simp only [setA, mem_union]; exact Or.inl (by simp)

lemma three_mem : (3 : ℕ) ∈ setA := by
  simp only [setA, mem_union]; exact Or.inl (by simp)

lemma stage_mem_setA {x k : ℕ} (hx : x ∈ stage k) : x ∈ setA := by
  simp only [setA, mem_union, mem_iUnion]
  exact Or.inr ⟨k, hx⟩

/-- The inherited "I" interval [2·Qk, 3·Qk] lives in A (as {2,3} for k=0, else Fk (k-1)). -/
lemma Imem (k : ℕ) : ∀ x, 2 * Q k ≤ x → x ≤ 3 * Q k → x ∈ setA := by
  rcases k with _ | k
  · intro x h1 h2
    have hq : Q 0 = 1 := by norm_num [Q]
    rw [hq] at h1 h2
    simp only [setA, mem_union, mem_insert_iff, mem_singleton_iff]
    exact Or.inl (by omega)
  · intro x h1 h2
    apply stage_mem_setA (k := k)
    simp only [stage, mem_union, Fk, mem_Icc]
    right
    rw [Q_succ] at h1 h2
    exact ⟨by omega, by omega⟩

lemma Bmem (k : ℕ) : ∀ x, 5 * Q k ≤ x → x ≤ 6 * Q k - 1 → x ∈ setA := by
  intro x h1 h2
  apply stage_mem_setA (k := k)
  simp only [stage, mem_union, Bk, mem_Icc]
  exact Or.inl (Or.inr ⟨h1, h2⟩)

lemma Fmem (k : ℕ) : ∀ x, 10 * Q k - 1 ≤ x → x ≤ 15 * Q k → x ∈ setA := by
  intro x h1 h2
  apply stage_mem_setA (k := k)
  simp only [stage, mem_union, Fk, mem_Icc]
  exact Or.inr ⟨h1, h2⟩

lemma ckmem (k : ℕ) : ∀ x, 4 * Q k ≤ x → x ≤ 4 * Q k → x ∈ setA := by
  intro x h1 h2
  have hx : x = ck k := by simp only [ck]; omega
  rw [hx]; exact ck_mem_setA k

/-- Cover a target n by a sum a+b with a ∈ [lo1,hi1], b ∈ [lo2,hi2], both in A. -/
lemma two_interval {lo1 hi1 lo2 hi2 n : ℕ}
    (memP : ∀ x, lo1 ≤ x → x ≤ hi1 → x ∈ setA)
    (memQ : ∀ x, lo2 ≤ x → x ≤ hi2 → x ∈ setA)
    (h1 : lo1 + lo2 ≤ n) (h2 : n ≤ hi1 + hi2) (h3 : lo1 ≤ hi1) (h4 : lo2 ≤ hi2) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  by_cases hc : n ≤ lo1 + hi2
  · exact ⟨lo1, memP lo1 le_rfl h3, n - lo1, memQ _ (by omega) (by omega), by omega⟩
  · exact ⟨n - hi2, memP _ (by omega) (by omega), hi2, memQ hi2 (by omega) le_rfl, by omega⟩

lemma basis_cover (k : ℕ) :
    ∀ n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  induction k with
  | zero =>
    intro n h4 hub
    have hq : Q 0 = 1 := by norm_num [Q]
    rw [hq] at hub
    interval_cases n
    · exact ⟨2, two_mem, 2, two_mem, rfl⟩
    · exact ⟨2, two_mem, 3, three_mem, rfl⟩
    · exact ⟨3, three_mem, 3, three_mem, rfl⟩
  | succ k ih =>
    intro n h4 hub
    have hQ := Q_pos k
    rw [Q_succ] at hub
    by_cases hsmall : n ≤ 6 * Q k
    · exact ih n h4 hsmall
    by_cases b1 : n ≤ 7 * Q k
    · exact two_interval (Imem k) (ckmem k) (by omega) (by omega) (by omega) (by omega)
    by_cases b2 : n ≤ 9 * Q k - 1
    · exact two_interval (Imem k) (Bmem k) (by omega) (by omega) (by omega) (by omega)
    by_cases b3 : n ≤ 10 * Q k - 1
    · exact two_interval (ckmem k) (Bmem k) (by omega) (by omega) (by omega) (by omega)
    by_cases b4 : n ≤ 12 * Q k - 2
    · exact two_interval (Bmem k) (Bmem k) (by omega) (by omega) (by omega) (by omega)
    by_cases b5 : n ≤ 18 * Q k
    · exact two_interval (Imem k) (Fmem k) (by omega) (by omega) (by omega) (by omega)
    by_cases b6 : n ≤ 21 * Q k - 1
    · exact two_interval (Bmem k) (Fmem k) (by omega) (by omega) (by omega) (by omega)
    · exact two_interval (Fmem k) (Fmem k) (by omega) (by omega) (by omega) (by omega)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn
    exact basis_cover n n hn (by have := Q_le n; have := Q_pos n; omega)
  · intro A₁ A₂ h1 h2 hcov hdisj hsyn
    obtain ⟨⟨C₁, hs1⟩, ⟨C₂, hs2⟩⟩ := hsyn
    set k := C₁ + C₂ + 1 with hk
    have hQk : k ≤ Q k := Q_le k
    have hckA : ck k ∈ setA := ck_mem_setA k
    rcases hcov (ck k) hckA with hcase | hcase
    · -- ck k ∈ A₁, so use gap on A₂
      have hnotA2 : ck k ∉ A₂ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hcase, hc⟩
        simp [hdisj] at hmem
      have hgap := gap_lem h2 hnotA2
      obtain ⟨m, hmS, hmI⟩ := hs2 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      rw [Set.eq_empty_iff_forall_notMem] at hgap
      exact hgap m ⟨hmJ, hmS⟩
    · -- ck k ∈ A₂, so use gap on A₁
      have hnotA1 : ck k ∉ A₁ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hc, hcase⟩
        simp [hdisj] at hmem
      have hgap := gap_lem h1 hnotA1
      obtain ⟨m, hmS, hmI⟩ := hs1 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      rw [Set.eq_empty_iff_forall_notMem] at hgap
      exact hgap m ⟨hmJ, hmS⟩

end Erdos741OAI

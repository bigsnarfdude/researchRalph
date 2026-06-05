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

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic Q lemmas -/

theorem Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

theorem Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

theorem Q_one_le (k : ℕ) : 1 ≤ Q k := Q_pos k

theorem n_le_Q : ∀ n, n ≤ Q n
  | 0 => by simp [Q]
  | (n+1) => by
      have ih := n_le_Q n
      have h5 : Q (n+1) = 5 * Q n := Q_succ n
      have h1 : 1 ≤ Q n := Q_one_le n
      omega

/-! ## Akn membership infrastructure -/

theorem akn_succ_subset (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  exact Set.mem_union_left _ hx

theorem ck_mem (k : ℕ) : ck k ∈ Akn (k+1) :=
  Set.mem_union_right _ (Set.mem_union_left _ (Set.mem_union_left _ rfl))

theorem Bk_sub (k : ℕ) : Bk k ⊆ Akn (k+1) := fun _ hx =>
  Set.mem_union_right _ (Set.mem_union_left _ (Set.mem_union_right _ hx))

theorem Fk_sub (k : ℕ) : Fk k ⊆ Akn (k+1) := fun _ hx =>
  Set.mem_union_right _ (Set.mem_union_right _ hx)

theorem I_mem : ∀ k, Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1)
  | 0 => by
      intro x hx
      simp only [mem_Icc, Q, pow_zero, mul_one] at hx
      have hmem : x ∈ ({2,3} : Set ℕ) := by
        simp only [Set.mem_insert_iff, Set.mem_singleton_iff]; omega
      exact Set.mem_union_left _ hmem
  | (k+1) => by
      intro x hx
      simp only [mem_Icc, Q_succ] at hx
      have hF : x ∈ Fk k := by
        simp only [Fk, mem_Icc]
        constructor <;> omega
      exact akn_succ_subset (k+1) (Fk_sub k hF)

theorem akn_subset_setA : ∀ k, Akn k ⊆ setA
  | 0 => by
      intro x hx
      exact Set.mem_union_left _ hx
  | (k+1) => by
      intro x hx
      rcases hx with hx | hx
      · exact akn_subset_setA k hx
      · exact Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, hx⟩)

/-! ## Coverage / basis lemmas -/

theorem cover_k (k : ℕ) : Icc (4 * Q k) (30 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  have hq : 1 ≤ Q k := Q_one_le k
  have mI : ∀ a, 2 * Q k ≤ a → a ≤ 3 * Q k → a ∈ Akn (k+1) :=
    fun a h1 h2 => I_mem k (mem_Icc.mpr ⟨h1, h2⟩)
  have mC : (4 * Q k) ∈ Akn (k+1) := ck_mem k
  have mB : ∀ a, 5 * Q k ≤ a → a ≤ 6 * Q k - 1 → a ∈ Akn (k+1) :=
    fun a h1 h2 => Bk_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  have mF : ∀ a, 10 * Q k - 1 ≤ a → a ≤ 15 * Q k → a ∈ Akn (k+1) :=
    fun a h1 h2 => Fk_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  by_cases h1 : x ≤ 5 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, mI _ (by omega) (by omega), x - 2 * Q k, mI _ (by omega) (by omega), by omega⟩
  by_cases h2 : x ≤ 6 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, mI _ (by omega) (by omega), x - 3 * Q k, mI _ (by omega) (by omega), by omega⟩
  by_cases h3 : x ≤ 7 * Q k
  · exact Set.mem_add.mpr ⟨4 * Q k, mC, x - 4 * Q k, mI _ (by omega) (by omega), by omega⟩
  by_cases h4 : x ≤ 8 * Q k
  · exact Set.mem_add.mpr ⟨5 * Q k, mB _ (by omega) (by omega), x - 5 * Q k, mI _ (by omega) (by omega), by omega⟩
  by_cases h5 : x ≤ 9 * Q k - 1
  · exact Set.mem_add.mpr ⟨3 * Q k, mI _ (by omega) (by omega), x - 3 * Q k, mB _ (by omega) (by omega), by omega⟩
  by_cases h6 : x ≤ 10 * Q k - 1
  · exact Set.mem_add.mpr ⟨4 * Q k, mC, x - 4 * Q k, mB _ (by omega) (by omega), by omega⟩
  by_cases h7 : x ≤ 11 * Q k - 1
  · exact Set.mem_add.mpr ⟨5 * Q k, mB _ (by omega) (by omega), x - 5 * Q k, mB _ (by omega) (by omega), by omega⟩
  by_cases h8 : x ≤ 12 * Q k - 2
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, mB _ (by omega) (by omega), x - (6 * Q k - 1), mB _ (by omega) (by omega), by omega⟩
  by_cases h9 : x ≤ 17 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, mI _ (by omega) (by omega), x - 2 * Q k, mF _ (by omega) (by omega), by omega⟩
  by_cases h10 : x ≤ 18 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, mI _ (by omega) (by omega), x - 3 * Q k, mF _ (by omega) (by omega), by omega⟩
  by_cases h11 : x ≤ 21 * Q k - 1
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, mB _ (by omega) (by omega), x - (6 * Q k - 1), mF _ (by omega) (by omega), by omega⟩
  by_cases h12 : x ≤ 25 * Q k - 1
  · exact Set.mem_add.mpr ⟨10 * Q k - 1, mF _ (by omega) (by omega), x - (10 * Q k - 1), mF _ (by omega) (by omega), by omega⟩
  · exact Set.mem_add.mpr ⟨15 * Q k, mF _ (by omega) (by omega), x - 15 * Q k, mF _ (by omega) (by omega), by omega⟩

theorem basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1)
  | 0 => by
      intro x hx
      simp only [mem_Icc, Q, pow_zero, mul_one] at hx
      obtain ⟨hlo, hhi⟩ := hx
      have h2 : (2:ℕ) ∈ Akn 1 := by
        have : (2:ℕ) ∈ ({2,3}:Set ℕ) := by simp
        exact Set.mem_union_left _ this
      have h3 : (3:ℕ) ∈ Akn 1 := by
        have : (3:ℕ) ∈ ({2,3}:Set ℕ) := by simp
        exact Set.mem_union_left _ this
      interval_cases x
      · exact Set.mem_add.mpr ⟨2, h2, 2, h2, by norm_num⟩
      · exact Set.mem_add.mpr ⟨2, h2, 3, h3, by norm_num⟩
      · exact Set.mem_add.mpr ⟨3, h3, 3, h3, by norm_num⟩
  | (k+1) => by
      intro x hx
      simp only [mem_Icc] at hx
      obtain ⟨hlo, hhi⟩ := hx
      by_cases hb : x ≤ 6 * Q k
      · have hmem : x ∈ Akn (k+1) + Akn (k+1) := basis_lem k (mem_Icc.mpr ⟨hlo, hb⟩)
        obtain ⟨a, ha, b, hbm, hab⟩ := Set.mem_add.mp hmem
        exact Set.mem_add.mpr ⟨a, akn_succ_subset (k+1) ha, b, akn_succ_subset (k+1) hbm, hab⟩
      · push_neg at hb
        rw [Q_succ k] at hhi
        have hx30 : x ≤ 30 * Q k := by omega
        have h4 : 4 * Q k ≤ x := by omega
        have hmem : x ∈ Akn (k+1) + Akn (k+1) := cover_k k (mem_Icc.mpr ⟨h4, hx30⟩)
        obtain ⟨a, ha, b, hbm, hab⟩ := Set.mem_add.mp hmem
        exact Set.mem_add.mpr ⟨a, akn_succ_subset (k+1) ha, b, akn_succ_subset (k+1) hbm, hab⟩

/-! ## Rigidity / gap infrastructure -/

theorem Q_le_of_le {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num) h

theorem stageval {x j : ℕ} (h : x ∈ stage j) :
    x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨ (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  simp only [stage, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at h
  tauto

theorem setA_cases {x : ℕ} (h : x ∈ setA) :
    (x = 2 ∨ x = 3) ∨ ∃ j, x ∈ stage j := by
  rcases h with h | h
  · left
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at h
    exact h
  · right
    exact Set.mem_iUnion.mp h

theorem setA_ge_two {x : ℕ} (h : x ∈ setA) : 2 ≤ x := by
  rcases setA_cases h with h23 | ⟨j, hj⟩
  · rcases h23 with h | h <;> omega
  · have hq := Q_one_le j
    rcases stageval hj with h | ⟨h, _⟩ | ⟨h, _⟩ <;> omega

theorem locate {x : ℕ} (hx : x ∈ setA) (k : ℕ) (hlt : x < 10 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ x = 10 * Q k - 1 := by
  have hq : 1 ≤ Q k := Q_one_le k
  rcases setA_cases hx with h23 | ⟨j, hj⟩
  · left; rcases h23 with h | h <;> omega
  · rcases lt_trichotomy j k with hlt' | heq | hgt
    · -- j < k : x ≤ 15 Q j ≤ 3 Q k
      have hub : x ≤ 15 * Q j := by
        rcases stageval hj with h | ⟨_, h2⟩ | ⟨_, h2⟩ <;> omega
      have h5 : 5 * Q j ≤ Q k := by
        have hh : Q (j+1) ≤ Q k := Q_le_of_le (by omega)
        rw [Q_succ] at hh; omega
      left; omega
    · -- j = k
      rw [heq] at hj
      rcases stageval hj with h | h | ⟨hlo, _⟩
      · right; left; exact h
      · right; right; left; exact h
      · right; right; right; omega
    · -- j > k : x ≥ 4 Q j ≥ 20 Q k, contradiction
      exfalso
      have hge : 5 * Q k ≤ Q j := by
        have hh : Q (k+1) ≤ Q j := Q_le_of_le (by omega)
        rw [Q_succ] at hh; omega
      have hxge : 4 * Q j ≤ x := by
        rcases stageval hj with h | ⟨h, _⟩ | ⟨h, _⟩ <;> omega
      omega

theorem rigidity {a b k : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hlo : 9 * Q k ≤ a + b) (hhi : a + b < 10 * Q k) :
    (a = 4 * Q k ∧ b ∈ Bk k) ∨ (b = 4 * Q k ∧ a ∈ Bk k) := by
  have hq : 1 ≤ Q k := Q_one_le k
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have haL : a < 10 * Q k := by omega
  have hbL : b < 10 * Q k := by omega
  have hca := locate ha k haL
  have hcb := locate hb k hbL
  rcases hca with hA | hB | hC | hD
  · exfalso; rcases hcb with h | h | h | h <;> omega
  · rcases hcb with h | h | h | h
    · exfalso; omega
    · exfalso; omega
    · left; exact ⟨hB, mem_Icc.mpr ⟨h.1, h.2⟩⟩
    · exfalso; omega
  · rcases hcb with h | h | h | h
    · exfalso; omega
    · right; exact ⟨h, mem_Icc.mpr ⟨hC.1, hC.2⟩⟩
    · exfalso; omega
    · exfalso; omega
  · exfalso; rcases hcb with h | h | h | h <;> omega

theorem gap_lem {T : Set ℕ} (hT : T ⊆ setA) {k : ℕ} (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hJ hadd
  simp only [Jk, mem_Ico] at hJ
  obtain ⟨hJlo, hJhi⟩ := hJ
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hadd
  have haA := hT ha
  have hbA := hT hb
  have hrig := rigidity (k := k) haA hbA (by omega) (by omega)
  rcases hrig with ⟨hac, _⟩ | ⟨hbc, _⟩
  · have heq : a = ck k := hac
    exact hck (heq ▸ ha)
  · have heq : b = ck k := hbc
    exact hck (heq ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · -- basis: every n ≥ 4 is a sum of two elements of setA
    intro n hn
    have hn6 : n ≤ 6 * Q n := le_trans (n_le_Q n) (by have := Q_one_le n; omega)
    have hmem : n ∈ Akn (n+1) + Akn (n+1) := basis_lem n (mem_Icc.mpr ⟨hn, hn6⟩)
    obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmem
    exact ⟨a, akn_subset_setA (n+1) ha, b, akn_subset_setA (n+1) hb, hab⟩
  · -- no partition is both-syndetic
    intro A₁ A₂ h1 h2 hpart hdisj
    rintro ⟨hs1, hs2⟩
    obtain ⟨C₁, hC₁⟩ := hs1
    obtain ⟨C₂, hC₂⟩ := hs2
    set C := max C₁ C₂ with hCdef
    set k := C + 1 with hk
    have hCk : C < Q k := by
      rw [hk]
      exact lt_of_lt_of_le (Nat.lt_succ_self C) (n_le_Q (C+1))
    have hckA : ck k ∈ setA := akn_subset_setA (k+1) (ck_mem k)
    rcases hpart (ck k) hckA with hck1 | hck2
    · -- ck k ∈ A₁ ⇒ ck k ∉ A₂
      have hckn2 : ck k ∉ A₂ := by
        intro h
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hck1, h⟩
        rw [hdisj] at hmem
        exact (Set.mem_empty_iff_false _).mp hmem
      have hgap := gap_lem h2 hckn2
      obtain ⟨m, hmAdd, hmIcc⟩ := hC₂ (9 * Q k)
      simp only [mem_Icc] at hmIcc
      have hC2C : C₂ ≤ C := le_max_right _ _
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]
        exact ⟨by omega, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmAdd⟩
      rw [hgap] at hcontra
      exact (Set.mem_empty_iff_false _).mp hcontra
    · -- ck k ∈ A₂ ⇒ ck k ∉ A₁
      have hckn1 : ck k ∉ A₁ := by
        intro h
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h, hck2⟩
        rw [hdisj] at hmem
        exact (Set.mem_empty_iff_false _).mp hmem
      have hgap := gap_lem h1 hckn1
      obtain ⟨m, hmAdd, hmIcc⟩ := hC₁ (9 * Q k)
      simp only [mem_Icc] at hmIcc
      have hC1C : C₁ ≤ C := le_max_left _ _
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]
        exact ⟨by omega, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmAdd⟩
      rw [hgap] at hcontra
      exact (Set.mem_empty_iff_false _).mp hcontra

end Erdos741OAI

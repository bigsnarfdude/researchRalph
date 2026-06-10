import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## Construction -/

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0     => {2, 3}
  | (k+1) => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma one_le_Q (k : ℕ) : 1 ≤ Q k := Q_pos k

/-! ## Monotonicity and containment in setA -/

lemma akn_subset_succ (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro y hy; exact Or.inl hy

lemma akn_mono {m n : ℕ} (h : m ≤ n) : Akn m ⊆ Akn n := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans (akn_subset_succ _)

lemma akn_sub_setA : ∀ k, Akn k ⊆ setA := by
  intro k
  induction k with
  | zero => intro y hy; exact Or.inl hy
  | succ k ih =>
    intro y hy
    rcases hy with hy | hy
    · exact ih hy
    · exact Or.inr (mem_iUnion.mpr ⟨k, hy⟩)

/-! ## The stage pieces live in `Akn (k+1)` -/

lemma ck_mem (k : ℕ) : ck k ∈ Akn (k + 1) := Or.inr (Or.inl (Or.inl rfl))

lemma Bk_sub (k : ℕ) : Bk k ⊆ Akn (k + 1) := fun _ hy => Or.inr (Or.inl (Or.inr hy))

lemma Fk_sub (k : ℕ) : Fk k ⊆ Akn (k + 1) := fun _ hy => Or.inr (Or.inr hy)

/-- The "I" interval `[2 Qk, 3 Qk]` sits in `Akn (k+1)`
(inherited from the previous stage's `Fk`, or `{2,3}` at the base). -/
lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro y hy
    simp only [Q, pow_zero, mul_one, mem_Icc] at hy
    refine Or.inl ?_
    show y ∈ ({2, 3} : Set ℕ)
    simp only [mem_insert_iff, mem_singleton_iff]
    omega
  | succ j =>
    intro y hy
    simp only [mem_Icc] at hy
    have hF : y ∈ Fk j := by
      simp only [Fk, mem_Icc]
      rw [Q_succ] at hy
      omega
    exact akn_mono (by omega) (Fk_sub j hF)

/-! ## Direct cover: `[4 Qk, 30 Qk]` is hit by sums from `Akn (k+1)` -/

lemma cover_lem (k : ℕ) :
    Icc (4 * Q k) (30 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  have hp := Q_pos k
  have hI : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k + 1) :=
    fun y h1 h2 => I_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  have hB : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k + 1) :=
    fun y h1 h2 => Bk_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  have hF : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k + 1) :=
    fun y h1 h2 => Fk_sub k (mem_Icc.mpr ⟨h1, h2⟩)
  have hc : (4 * Q k) ∈ Akn (k + 1) := ck_mem k
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  rw [Set.mem_add]
  by_cases c1 : x ≤ 5 * Q k
  · exact ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases c2 : x ≤ 6 * Q k
  · exact ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases c3 : x ≤ 7 * Q k
  · exact ⟨4 * Q k, hc, x - 4 * Q k, hI _ (by omega) (by omega), by omega⟩
  by_cases c4 : x ≤ 8 * Q k - 1
  · exact ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases c5 : x ≤ 9 * Q k - 1
  · exact ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases c6 : x ≤ 10 * Q k - 1
  · exact ⟨4 * Q k, hc, x - 4 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases c7 : x ≤ 11 * Q k - 1
  · exact ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k, hB _ (by omega) (by omega), by omega⟩
  by_cases c8 : x ≤ 12 * Q k - 2
  · exact ⟨6 * Q k - 1, hB _ (by omega) (by omega), x - (6 * Q k - 1), hB _ (by omega) (by omega), by omega⟩
  by_cases c9 : x ≤ 17 * Q k
  · exact ⟨2 * Q k, hI _ (by omega) (by omega), x - 2 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases c10 : x ≤ 18 * Q k
  · exact ⟨3 * Q k, hI _ (by omega) (by omega), x - 3 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases c11 : x ≤ 20 * Q k
  · exact ⟨5 * Q k, hB _ (by omega) (by omega), x - 5 * Q k, hF _ (by omega) (by omega), by omega⟩
  by_cases c12 : x ≤ 21 * Q k - 1
  · exact ⟨6 * Q k - 1, hB _ (by omega) (by omega), x - (6 * Q k - 1), hF _ (by omega) (by omega), by omega⟩
  by_cases c13 : x ≤ 25 * Q k - 1
  · exact ⟨10 * Q k - 1, hF _ (by omega) (by omega), x - (10 * Q k - 1), hF _ (by omega) (by omega), by omega⟩
  · exact ⟨15 * Q k, hF _ (by omega) (by omega), x - 15 * Q k, hF _ (by omega) (by omega), by omega⟩

/-! ## Basis: `[4, 6 Qk]` is covered, by induction -/

lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    exact cover_lem 0 (by simp only [Q, pow_zero, mul_one, mem_Icc]; omega)
  | succ k ih =>
    intro x hx
    have hsucc : Q (k + 1) = 5 * Q k := Q_succ k
    have hp := Q_pos k
    simp only [mem_Icc] at hx
    by_cases hsmall : x ≤ 6 * Q k
    · have hmem : x ∈ Akn (k + 1) + Akn (k + 1) :=
        ih (mem_Icc.mpr ⟨by omega, by omega⟩)
      exact Set.add_subset_add (akn_subset_succ _) (akn_subset_succ _) hmem
    · have hmem : x ∈ Akn (k + 1) + Akn (k + 1) :=
        cover_lem k (mem_Icc.mpr ⟨by omega, by omega⟩)
      exact Set.add_subset_add (akn_subset_succ _) (akn_subset_succ _) hmem

/-! ## Growth facts -/

lemma Q_mono {i j : ℕ} (h : i ≤ j) : Q i ≤ Q j :=
  Nat.pow_le_pow_right (by norm_num) h

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ k ih =>
    have hs : Q (k + 1) = 5 * Q k := Q_succ k
    have hp := Q_pos k
    omega

/-! ## Element structure of `setA` -/

lemma elt_cases {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j,
      x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨ (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  rcases hx with h | h
  · simp only [mem_insert_iff, mem_singleton_iff] at h
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
  · rw [mem_iUnion] at h
    obtain ⟨j, hj⟩ := h
    refine Or.inr (Or.inr ⟨j, ?_⟩)
    simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hj
    rcases hj with (hj | hj) | hj
    · exact Or.inl hj
    · exact Or.inr (Or.inl hj)
    · exact Or.inr (Or.inr hj)

lemma two_le {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases elt_cases hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have := Q_pos j; rcases hj with h | h | h <;> omega

/-- For `a ∈ setA` with `a < 10 Qk`, locate `a`: either `≤ 3 Qk`, or in one of the
stage-`k` pieces. -/
lemma elt_struct (k a : ℕ) (ha : a ∈ setA) (hlt : a < 10 * Q k) :
    a ≤ 3 * Q k ∨ a = 4 * Q k ∨ (5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1)
      ∨ (10 * Q k - 1 ≤ a ∧ a ≤ 15 * Q k) := by
  have hpk := Q_pos k
  rcases elt_cases ha with h | h | ⟨j, hj⟩
  · left; omega
  · left; omega
  · have hpj := Q_pos j
    have hge : 4 * Q j ≤ a := by rcases hj with h | h | h <;> omega
    have hjk : j ≤ k := by
      by_contra hcon
      push_neg at hcon
      have hm : Q (k + 1) ≤ Q j := Q_mono hcon
      have hs : Q (k + 1) = 5 * Q k := Q_succ k
      omega
    rcases hjk.lt_or_eq with hlt2 | heq
    · have hle15 : a ≤ 15 * Q j := by rcases hj with h | h | h <;> omega
      have h5 : 5 * Q j ≤ Q k := by
        have hm : Q (j + 1) ≤ Q k := Q_mono hlt2
        have hs : Q (j + 1) = 5 * Q j := Q_succ j
        omega
      left; omega
    · rw [heq] at hj
      right; exact hj

/-! ## Rigidity: only `ck k + Bk k` sums land in `[9 Qk, 10 Qk)` -/

lemma rigidity (k n a b : ℕ) (hn1 : 9 * Q k ≤ n) (hn2 : n < 10 * Q k)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hpk := Q_pos k
  have ha2 := two_le ha
  have hb2 := two_le hb
  have hsa := elt_struct k a ha (by omega)
  have hsb := elt_struct k b hb (by omega)
  rcases hsa with hA | hA | hA | hA <;> rcases hsb with hB | hB | hB | hB <;>
    first
      | exact Or.inl ⟨by show a = 4 * Q k; omega,
          by show b ∈ Icc (5 * Q k) (6 * Q k - 1); exact mem_Icc.mpr ⟨by omega, by omega⟩⟩
      | exact Or.inr ⟨by show b = 4 * Q k; omega,
          by show a ∈ Icc (5 * Q k) (6 * Q k - 1); exact mem_Icc.mpr ⟨by omega, by omega⟩⟩
      | (exfalso; omega)

/-! ## Gap lemma: if `ck k ∉ T` then `T + T` misses `Jk k` -/

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n, n ∈ Jk k → n ∉ T + T := by
  intro n hn hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  simp only [Jk, mem_Ico] at hn
  rcases rigidity k n a b hn.1 hn.2 (hT ha) (hT hb) hab with ⟨h1, _⟩ | ⟨h1, _⟩
  · exact hck (h1 ▸ ha)
  · exact hck (h1 ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · -- basis
    intro n hn
    have hmem : n ∈ Akn (n + 1) + Akn (n + 1) := by
      apply basis_lem n
      simp only [mem_Icc]
      refine ⟨hn, ?_⟩
      have := n_le_Q n
      have := Q_pos n
      omega
    rw [Set.mem_add] at hmem
    obtain ⟨a, ha, b, hb, hab⟩ := hmem
    exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩
  · -- no partition is both-syndetic
    rintro A₁ A₂ hA1 hA2 hcover hdisj ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    set k := max C₁ C₂ + 1 with hk
    have hkle : k ≤ Q k := n_le_Q k
    have hQ1 : C₁ < Q k := by
      have : C₁ ≤ max C₁ C₂ := le_max_left _ _; omega
    have hQ2 : C₂ < Q k := by
      have : C₂ ≤ max C₁ C₂ := le_max_right _ _; omega
    have hck_mem : ck k ∈ setA := akn_sub_setA (k + 1) (ck_mem k)
    rcases hcover _ hck_mem with hc1 | hc2
    · -- ck k ∈ A₁, so ck k ∉ A₂; A₂ + A₂ must hit Jk k — contradiction
      have hnotA2 : ck k ∉ A₂ := by
        intro h
        have hh : ck k ∈ A₁ ∩ A₂ := ⟨hc1, h⟩
        rw [hdisj] at hh; simp at hh
      obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; exact ⟨hmI.1, by omega⟩
      exact gap_lem k A₂ hA2 hnotA2 m hmJ hmS
    · -- ck k ∈ A₂, so ck k ∉ A₁; A₁ + A₁ must hit Jk k — contradiction
      have hnotA1 : ck k ∉ A₁ := by
        intro h
        have hh : ck k ∈ A₁ ∩ A₂ := ⟨h, hc2⟩
        rw [hdisj] at hh; simp at hh
      obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
      simp only [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; exact ⟨hmI.1, by omega⟩
      exact gap_lem k A₁ hA1 hnotA1 m hmJ hmS

end Erdos741OAI

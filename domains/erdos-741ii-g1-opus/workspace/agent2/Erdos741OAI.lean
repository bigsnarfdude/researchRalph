import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## The construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k + 1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Akn_zero : Akn 0 = {2, 3} := rfl
lemma Akn_succ (k : ℕ) : Akn (k + 1) = Akn k ∪ {ck k} ∪ Bk k ∪ Fk k := rfl

/-! ## Arithmetic on Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by unfold Q; rw [pow_succ]; ring

lemma Q_le {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma n_lt_Q (n : ℕ) : n < Q n := by
  induction n with
  | zero => exact Q_pos 0
  | succ m ih =>
    have h1 : 0 < Q m := Q_pos m
    have h2 : Q (m + 1) = 5 * Q m := Q_succ m
    omega

/-! ## Akn membership helpers -/

lemma ck_mem (k : ℕ) : ck k ∈ Akn (k + 1) := by
  rw [Akn_succ]
  exact mem_union_left _ (mem_union_left _ (mem_union_right _ (mem_singleton_iff.mpr rfl)))

lemma Bk_sub (k : ℕ) : Bk k ⊆ Akn (k + 1) := by
  intro x hx; rw [Akn_succ]
  exact mem_union_left _ (mem_union_right _ hx)

lemma Fk_sub (k : ℕ) : Fk k ⊆ Akn (k + 1) := by
  intro x hx; rw [Akn_succ]
  exact mem_union_right _ hx

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx; rw [Akn_succ]
  exact mem_union_left _ (mem_union_left _ (mem_union_left _ hx))

lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    rw [Akn_succ, Akn_zero]
    have hx23 : x = 2 ∨ x = 3 := by omega
    rcases hx23 with h | h <;> subst h
    · exact mem_union_left _ (mem_union_left _ (mem_union_left _ (mem_insert _ _)))
    · exact mem_union_left _ (mem_union_left _ (mem_union_left _ (mem_insert_of_mem _ rfl)))
  | succ l =>
    intro x hx
    apply akn_mono (l + 1)
    rw [Akn_succ]
    have hs := Q_succ l
    simp only [mem_Icc] at hx
    have hF : x ∈ Fk l := by simp only [Fk, mem_Icc]; omega
    exact mem_union_right _ hF

lemma stage_sub (j : ℕ) : {ck j} ∪ Bk j ∪ Fk j ⊆ setA := by
  intro x hx
  show x ∈ {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)
  exact mem_union_right _ (mem_iUnion.mpr ⟨j, hx⟩)

lemma akn_sub (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero =>
    intro x hx
    rw [Akn_zero] at hx
    show x ∈ {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)
    exact mem_union_left _ hx
  | succ l ih =>
    intro x hx
    rw [Akn_succ] at hx
    simp only [mem_union] at hx
    rcases hx with ((h | h) | h) | h
    · exact ih h
    · exact stage_sub l (mem_union_left _ (mem_union_left _ h))
    · exact stage_sub l (mem_union_left _ (mem_union_right _ h))
    · exact stage_sub l (mem_union_right _ h)

/-! ## Basis property -/

lemma basis_lem : ∀ k n, 4 ≤ n → n ≤ 6 * Q k →
    ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = n := by
  intro k
  induction k with
  | zero =>
    intro n h4 h6
    simp only [Q, pow_zero, mul_one] at h6
    have h2 : (2 : ℕ) ∈ Akn 1 := by
      rw [Akn_succ, Akn_zero]
      exact mem_union_left _ (mem_union_left _ (mem_union_left _ (mem_insert _ _)))
    have h3 : (3 : ℕ) ∈ Akn 1 := by
      rw [Akn_succ, Akn_zero]
      exact mem_union_left _ (mem_union_left _ (mem_union_left _ (mem_insert_of_mem _ rfl)))
    interval_cases n
    · exact ⟨2, h2, 2, h2, rfl⟩
    · exact ⟨2, h2, 3, h3, rfl⟩
    · exact ⟨3, h3, 3, h3, rfl⟩
  | succ l ih =>
    intro n h4 h6
    have hs := Q_succ l
    have hQ1 := Q_pos l
    by_cases hsmall : n ≤ 6 * Q l
    · obtain ⟨a, ha, b, hb, hab⟩ := ih n h4 hsmall
      exact ⟨a, akn_mono (l + 1) ha, b, akn_mono (l + 1) hb, hab⟩
    · push_neg at hsmall
      have h6' : n ≤ 30 * Q l := by omega
      have hck : ck l = 4 * Q l := rfl
      have inI : ∀ x, 2 * Q l ≤ x → x ≤ 3 * Q l → x ∈ Akn (l + 2) :=
        fun x p1 p2 => akn_mono (l + 1) (I_sub l (mem_Icc.mpr ⟨p1, p2⟩))
      have inB : ∀ x, 5 * Q l ≤ x → x ≤ 6 * Q l - 1 → x ∈ Akn (l + 2) :=
        fun x p1 p2 => akn_mono (l + 1) (Bk_sub l (by simp only [Bk, mem_Icc]; exact ⟨p1, p2⟩))
      have inF : ∀ x, 10 * Q l - 1 ≤ x → x ≤ 15 * Q l → x ∈ Akn (l + 2) :=
        fun x p1 p2 => akn_mono (l + 1) (Fk_sub l (by simp only [Fk, mem_Icc]; exact ⟨p1, p2⟩))
      have ckmem : ck l ∈ Akn (l + 2) := akn_mono (l + 1) (ck_mem l)
      rcases le_or_gt n (7 * Q l) with h | hA
      · exact ⟨ck l, ckmem, n - 4 * Q l, inI _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (8 * Q l) with h | hB
      · exact ⟨5 * Q l, inB _ (by omega) (by omega), n - 5 * Q l, inI _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (9 * Q l - 1) with h | hC
      · exact ⟨6 * Q l - 1, inB _ (by omega) (by omega), n - (6 * Q l - 1), inI _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (10 * Q l - 1) with h | hD
      · exact ⟨ck l, ckmem, n - 4 * Q l, inB _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (11 * Q l - 1) with h | hE
      · exact ⟨5 * Q l, inB _ (by omega) (by omega), n - 5 * Q l, inB _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (12 * Q l - 2) with h | hF2
      · exact ⟨6 * Q l - 1, inB _ (by omega) (by omega), n - (6 * Q l - 1), inB _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (17 * Q l) with h | hG
      · exact ⟨n - 2 * Q l, inF _ (by omega) (by omega), 2 * Q l, inI _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (18 * Q l) with h | hH
      · exact ⟨n - 3 * Q l, inF _ (by omega) (by omega), 3 * Q l, inI _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (20 * Q l) with h | hI2
      · exact ⟨n - 5 * Q l, inF _ (by omega) (by omega), 5 * Q l, inB _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (21 * Q l - 1) with h | hJ
      · exact ⟨n - (6 * Q l - 1), inF _ (by omega) (by omega), 6 * Q l - 1, inB _ (by omega) (by omega), by omega⟩
      rcases le_or_gt n (25 * Q l - 1) with h | hK
      · exact ⟨n - (10 * Q l - 1), inF _ (by omega) (by omega), 10 * Q l - 1, inF _ (by omega) (by omega), by omega⟩
      · exact ⟨n - 15 * Q l, inF _ (by omega) (by omega), 15 * Q l, inF _ (by omega) (by omega), by omega⟩

/-! ## Rigidity and gap -/

lemma ge_two : ∀ a ∈ setA, 2 ≤ a := by
  intro a ha
  simp only [setA, mem_union, mem_iUnion] at ha
  rcases ha with h | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at h; omega
  · simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hj
    have hp := Q_pos j
    rcases hj with (h | h) | h <;> omega

lemma elt_bound (k : ℕ) : ∀ a ∈ setA,
    a = 4 * Q k ∨ (5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1) ∨ a ≤ 3 * Q k ∨ 10 * Q k - 1 ≤ a := by
  intro a ha
  simp only [setA, mem_union, mem_iUnion] at ha
  rcases ha with h | ⟨j, hj⟩
  · simp only [mem_insert_iff, mem_singleton_iff] at h
    have hp := Q_pos k
    omega
  · simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hj
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have hjk : 5 * Q j ≤ Q k := by
        have h' : Q (j + 1) ≤ Q k := Q_le (by omega)
        rwa [Q_succ] at h'
      rcases hj with (h | h) | h <;> omega
    · rw [hje] at hj
      rcases hj with (h | h) | h <;> omega
    · have hkj : 5 * Q k ≤ Q j := by
        have h' : Q (k + 1) ≤ Q j := Q_le (by omega)
        rwa [Q_succ] at h'
      rcases hj with (h | h) | h <;> omega

lemma rigidity (k : ℕ) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b ∈ Ico (9 * Q k) (10 * Q k)) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [mem_Ico] at hab
  obtain ⟨hlo, hhi⟩ := hab
  have ha2 := ge_two a ha
  have hb2 := ge_two b hb
  have hbA := elt_bound k a ha
  have hbB := elt_bound k b hb
  have hQ := Q_pos k
  simp only [ck, Bk, mem_Icc]
  omega

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T)
    (n : ℕ) (hn : n ∈ Ico (9 * Q k) (10 * Q k)) :
    ¬ ∃ a ∈ T, ∃ b ∈ T, a + b = n := by
  rintro ⟨a, haT, b, hbT, hab⟩
  have ha := hT haT
  have hb := hT hbT
  have hrig := rigidity k a b ha hb (by rw [hab]; exact hn)
  rcases hrig with ⟨hac, _⟩ | ⟨hbc, _⟩
  · exact hck (hac ▸ haT)
  · exact hck (hbc ▸ hbT)

/-! ## Main theorem -/

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
    have hk : n ≤ 6 * Q n := by
      have hlt := n_lt_Q n; omega
    obtain ⟨a, ha, b, hb, hab⟩ := basis_lem n n hn hk
    exact ⟨a, akn_sub _ ha, b, akn_sub _ hb, hab⟩
  · intro A₁ A₂ hA1 hA2 hcover hdisj
    rintro ⟨hsyn1, hsyn2⟩
    obtain ⟨C₁, hC1⟩ := hsyn1
    obtain ⟨C₂, hC2⟩ := hsyn2
    set k := C₁ + C₂ + 1 with hk_def
    have hQk : C₁ + C₂ < Q k := by
      have hlt := n_lt_Q k; omega
    have hckA : ck k ∈ setA := by
      apply stage_sub k
      exact mem_union_left _ (mem_union_left _ (mem_singleton_iff.mpr rfl))
    rcases hcover (ck k) hckA with h1 | h2
    · have hnotA2 : ck k ∉ A₂ := by
        intro hm
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h1, hm⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmMem, hmIcc⟩ := hC2 (9 * Q k)
      simp only [mem_Icc] at hmIcc
      rw [Set.mem_add] at hmMem
      have hmIco : m ∈ Ico (9 * Q k) (10 * Q k) := by
        simp only [mem_Ico]; omega
      exact gap_lem k A₂ hA2 hnotA2 m hmIco hmMem
    · have hnotA1 : ck k ∉ A₁ := by
        intro hm
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hm, h2⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmMem, hmIcc⟩ := hC1 (9 * Q k)
      simp only [mem_Icc] at hmIcc
      rw [Set.mem_add] at hmMem
      have hmIco : m ∈ Ico (9 * Q k) (10 * Q k) := by
        simp only [mem_Ico]; omega
      exact gap_lem k A₁ hA1 hnotA1 m hmIco hmMem

end Erdos741OAI

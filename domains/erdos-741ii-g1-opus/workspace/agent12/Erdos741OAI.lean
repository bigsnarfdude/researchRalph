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
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ stage k

/-! ## Arithmetic on Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp only [Q, pow_succ]; ring

lemma Q_mono {i j : ℕ} (h : i ≤ j) : Q i ≤ Q j := by
  simp only [Q]; exact Nat.pow_le_pow_right (by norm_num) h

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ m ih =>
    have h := Q_succ m
    have hp := Q_pos m
    omega

/-! ## Akn machinery -/

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inl hx

lemma stage_sub (k : ℕ) : stage k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inr hx

lemma akn_subset_A (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero =>
    intro x hx
    simp only [Akn] at hx
    exact Or.inl hx
  | succ k ih =>
    intro x hx
    simp only [Akn, mem_union] at hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Or.inr (mem_iUnion.mpr ⟨k, hx⟩)

/-! ## Membership classification and stage bounds -/

lemma mem_setA {x : ℕ} (hx : x ∈ setA) : (2 ≤ x ∧ x ≤ 3) ∨ ∃ i, x ∈ stage i := by
  simp only [setA, mem_union, mem_iUnion, Set.mem_insert_iff, Set.mem_singleton_iff] at hx
  rcases hx with (h | h) | ⟨i, hi⟩
  · left; omega
  · left; omega
  · right; exact ⟨i, hi⟩

lemma stage_ge {i x : ℕ} (hx : x ∈ stage i) : 4 * Q i ≤ x := by
  have := Q_pos i
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hx
  rcases hx with (h | h) | h <;> omega

lemma stage_le {i x : ℕ} (hx : x ∈ stage i) : x ≤ 15 * Q i := by
  have := Q_pos i
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hx
  rcases hx with (h | h) | h <;> omega

lemma stage_lt_bound {j k : ℕ} (h : j < k) : 15 * Q j ≤ 3 * Q k := by
  have h1 : Q (j+1) ≤ Q k := Q_mono (by omega)
  rw [Q_succ] at h1
  omega

lemma stage_gt_bound {k j : ℕ} (h : k < j) : 20 * Q k ≤ 4 * Q j := by
  have h1 : Q (k+1) ≤ Q j := Q_mono (by omega)
  rw [Q_succ] at h1
  omega

/-! ## The interval I and the basis -/

lemma hI (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1) := by
  cases k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    have h0 : Q 0 = 1 := rfl
    rw [h0] at hx
    simp only [Akn, mem_union]
    left
    show x ∈ ({2, 3} : Set ℕ)
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    omega
  | succ m =>
    intro x hx
    rw [mem_Icc, Q_succ m] at hx
    apply akn_mono (m+1)
    apply stage_sub m
    show x ∈ ({ck m} ∪ Bk m) ∪ Fk m
    refine Or.inr ?_
    simp only [Fk, mem_Icc]
    omega

lemma two_mem_Akn1 : (2 : ℕ) ∈ Akn 1 := by
  simp only [Akn, mem_union]
  left
  show (2 : ℕ) ∈ ({2, 3} : Set ℕ)
  simp

lemma three_mem_Akn1 : (3 : ℕ) ∈ Akn 1 := by
  simp only [Akn, mem_union]
  left
  show (3 : ℕ) ∈ ({2, 3} : Set ℕ)
  simp

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    have h0 : Q 0 = 1 := rfl
    rw [h0] at hx
    obtain ⟨hx1, hx2⟩ := hx
    interval_cases x
    · exact Set.add_mem_add two_mem_Akn1 two_mem_Akn1
    · exact Set.add_mem_add two_mem_Akn1 three_mem_Akn1
    · exact Set.add_mem_add three_mem_Akn1 three_mem_Akn1
  | succ k ih =>
    intro x hx
    rw [mem_Icc, Q_succ k] at hx
    obtain ⟨hx1, hx2⟩ := hx
    have hq := Q_pos k
    by_cases hsmall : x ≤ 6 * Q k
    · have hmono : Akn (k+1) + Akn (k+1) ⊆ Akn (k+1+1) + Akn (k+1+1) :=
        Set.add_subset_add (akn_mono (k+1)) (akn_mono (k+1))
      exact hmono (ih (mem_Icc.mpr ⟨hx1, hsmall⟩))
    · push_neg at hsmall
      have memc : (4 * Q k) ∈ Akn (k+1+1) := by
        apply akn_mono (k+1); apply stage_sub k
        show (4 * Q k) ∈ ({ck k} ∪ Bk k) ∪ Fk k
        exact Or.inl (Or.inl (by simp [ck]))
      have memI : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k+1+1) := by
        intro y h1 h2; exact akn_mono (k+1) (hI k (mem_Icc.mpr ⟨h1, h2⟩))
      have memB : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k+1+1) := by
        intro y h1 h2
        apply akn_mono (k+1); apply stage_sub k
        show y ∈ ({ck k} ∪ Bk k) ∪ Fk k
        exact Or.inl (Or.inr (mem_Icc.mpr ⟨h1, h2⟩))
      have memF : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k+1+1) := by
        intro y h1 h2
        apply akn_mono (k+1); apply stage_sub k
        show y ∈ ({ck k} ∪ Bk k) ∪ Fk k
        exact Or.inr (mem_Icc.mpr ⟨h1, h2⟩)
      by_cases r1 : x ≤ 7 * Q k
      · rw [show x = (x - 4 * Q k) + 4 * Q k by omega]
        exact Set.add_mem_add (memI _ (by omega) (by omega)) memc
      by_cases r2 : x ≤ 9 * Q k - 1
      · by_cases r2a : x ≤ 8 * Q k
        · rw [show x = (x - 5 * Q k) + 5 * Q k by omega]
          exact Set.add_mem_add (memI _ (by omega) (by omega)) (memB _ (by omega) (by omega))
        · rw [show x = 3 * Q k + (x - 3 * Q k) by omega]
          exact Set.add_mem_add (memI _ (by omega) (by omega)) (memB _ (by omega) (by omega))
      by_cases r3 : x ≤ 10 * Q k - 1
      · rw [show x = 4 * Q k + (x - 4 * Q k) by omega]
        exact Set.add_mem_add memc (memB _ (by omega) (by omega))
      by_cases r4 : x ≤ 12 * Q k - 2
      · by_cases r4a : x ≤ 11 * Q k - 1
        · rw [show x = 5 * Q k + (x - 5 * Q k) by omega]
          exact Set.add_mem_add (memB _ (by omega) (by omega)) (memB _ (by omega) (by omega))
        · rw [show x = (6 * Q k - 1) + (x - (6 * Q k - 1)) by omega]
          exact Set.add_mem_add (memB _ (by omega) (by omega)) (memB _ (by omega) (by omega))
      by_cases r5 : x ≤ 18 * Q k
      · by_cases r5a : x ≤ 15 * Q k
        · rw [show x = 2 * Q k + (x - 2 * Q k) by omega]
          exact Set.add_mem_add (memI _ (by omega) (by omega)) (memF _ (by omega) (by omega))
        · rw [show x = 3 * Q k + (x - 3 * Q k) by omega]
          exact Set.add_mem_add (memI _ (by omega) (by omega)) (memF _ (by omega) (by omega))
      by_cases r6 : x ≤ 21 * Q k - 1
      · by_cases r6a : x ≤ 20 * Q k
        · rw [show x = 5 * Q k + (x - 5 * Q k) by omega]
          exact Set.add_mem_add (memB _ (by omega) (by omega)) (memF _ (by omega) (by omega))
        · rw [show x = (6 * Q k - 1) + (x - (6 * Q k - 1)) by omega]
          exact Set.add_mem_add (memB _ (by omega) (by omega)) (memF _ (by omega) (by omega))
      · by_cases r7a : x ≤ 25 * Q k - 1
        · rw [show x = (10 * Q k - 1) + (x - (10 * Q k - 1)) by omega]
          exact Set.add_mem_add (memF _ (by omega) (by omega)) (memF _ (by omega) (by omega))
        · rw [show x = 15 * Q k + (x - 15 * Q k) by omega]
          exact Set.add_mem_add (memF _ (by omega) (by omega)) (memF _ (by omega) (by omega))

/-! ## Rigidity -/

lemma rigidity (k a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hn : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hq := Q_pos k
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hlo, hhi⟩ := hn
  rcases mem_setA ha with ha3 | ⟨i, hai⟩
  · rcases mem_setA hb with hb3 | ⟨j, hbj⟩
    · exfalso; omega
    · exfalso
      have hbge := stage_ge hbj
      have hble := stage_le hbj
      have hqj := Q_pos j
      rcases lt_trichotomy j k with hjlt | hjeq | hjgt
      · have := stage_lt_bound hjlt
        omega
      · rw [hjeq] at hbj
        simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hbj
        rcases hbj with (h | h) | h <;> omega
      · have := stage_gt_bound hjgt
        omega
  · rcases mem_setA hb with hb3 | ⟨j, hbj⟩
    · exfalso
      have hage := stage_ge hai
      have hale := stage_le hai
      have hqi := Q_pos i
      rcases lt_trichotomy i k with hilt | hieq | higt
      · have := stage_lt_bound hilt
        omega
      · rw [hieq] at hai
        simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hai
        rcases hai with (h | h) | h <;> omega
      · have := stage_gt_bound higt
        omega
    · have hage := stage_ge hai
      have hale := stage_le hai
      have hbge := stage_ge hbj
      have hble := stage_le hbj
      have hqi := Q_pos i
      have hqj := Q_pos j
      have hik_le : i ≤ k := by
        by_contra h; push_neg at h
        have hb2 := stage_gt_bound h
        omega
      have hjk_le : j ≤ k := by
        by_contra h; push_neg at h
        have hb2 := stage_gt_bound h
        omega
      by_cases hik : i = k
      · rw [hik] at hai
        simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hai
        rcases hai with (h | h) | h
        · left
          refine ⟨by show a = 4 * Q k; exact h, ?_⟩
          show b ∈ Icc (5 * Q k) (6 * Q k - 1)
          exact mem_Icc.mpr ⟨by omega, by omega⟩
        · by_cases hjk : j = k
          · rw [hjk] at hbj
            simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hbj
            rcases hbj with (hb' | hb') | hb'
            · right
              refine ⟨by show b = 4 * Q k; exact hb', ?_⟩
              show a ∈ Icc (5 * Q k) (6 * Q k - 1)
              exact mem_Icc.mpr ⟨by omega, by omega⟩
            · exfalso; omega
            · exfalso; omega
          · exfalso
            have hjlt : j < k := lt_of_le_of_ne hjk_le hjk
            have := stage_lt_bound hjlt
            omega
        · exfalso; omega
      · exfalso
        have hilt : i < k := lt_of_le_of_ne hik_le hik
        have hib := stage_lt_bound hilt
        by_cases hjk : j = k
        · rw [hjk] at hbj
          simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hbj
          rcases hbj with (hb' | hb') | hb' <;> omega
        · have hjlt : j < k := lt_of_le_of_ne hjk_le hjk
          have := stage_lt_bound hjlt
          omega

/-! ## Gap lemma -/

lemma gap (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false, not_and]
  intro hJ hadd
  rw [Set.mem_add] at hadd
  obtain ⟨a, ha, b, hb, hab⟩ := hadd
  have hrig := rigidity k a b (hT ha) (hT hb) (by rw [hab]; exact hJ)
  rcases hrig with ⟨ha', _⟩ | ⟨hb', _⟩
  · exact hck (ha' ▸ ha)
  · exact hck (hb' ▸ hb)

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
    have hmem : n ∈ Akn (n+1) + Akn (n+1) := by
      apply basis_lem n
      rw [mem_Icc]
      have hle := n_le_Q n
      have hp := Q_pos n
      exact ⟨hn, by omega⟩
    rw [Set.mem_add] at hmem
    obtain ⟨a, ha, b, hb, hab⟩ := hmem
    exact ⟨a, akn_subset_A _ ha, b, akn_subset_A _ hb, hab⟩
  · intro A₁ A₂ hA1 hA2 hcover hdisj
    rintro ⟨⟨C₁, hsyn1⟩, ⟨C₂, hsyn2⟩⟩
    set k := max C₁ C₂ + 1 with hk
    have hQk1 : C₁ < Q k := by
      have h1 : k ≤ Q k := n_le_Q k
      omega
    have hQk2 : C₂ < Q k := by
      have h1 : k ≤ Q k := n_le_Q k
      omega
    have hck_inA : (ck k) ∈ setA := by
      have hm : ck k ∈ ⋃ i, stage i := mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩
      exact Or.inr hm
    rcases hcover _ hck_inA with hA | hA
    · have hnot : ck k ∉ A₂ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hA, hc⟩
        rw [hdisj] at this; exact this
      have hgap := gap k A₂ hA2 hnot
      obtain ⟨m, hmS, hmI⟩ := hsyn2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        show m ∈ Ico (9 * Q k) (10 * Q k)
        rw [mem_Ico]; exact ⟨by omega, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra; exact hcontra
    · have hnot : ck k ∉ A₁ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hc, hA⟩
        rw [hdisj] at this; exact this
      have hgap := gap k A₁ hA1 hnot
      obtain ⟨m, hmS, hmI⟩ := hsyn1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        show m ∈ Ico (9 * Q k) (10 * Q k)
        rw [mem_Ico]; exact ⟨by omega, by omega⟩
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra; exact hcontra

end Erdos741OAI

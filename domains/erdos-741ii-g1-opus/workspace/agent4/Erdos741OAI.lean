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

/-! ## Basic arithmetic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma n_lt_Q : ∀ n, n < Q n := by
  intro n
  induction n with
  | zero => unfold Q; norm_num
  | succ m ih =>
      have hs : Q (m + 1) = 5 * Q m := Q_succ m
      have hm := Q_pos m
      omega

/-! ## Membership helpers -/

lemma two_mem : (2 : ℕ) ∈ setA := by
  simp only [setA]; exact Set.mem_union_left _ (by simp)

lemma three_mem : (3 : ℕ) ∈ setA := by
  simp only [setA]; exact Set.mem_union_left _ (by simp)

lemma ck_mem (k : ℕ) : 4 * Q k ∈ setA := by
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨k, Or.inl (Or.inl rfl)⟩

lemma Bk_sub_setA (k : ℕ) : Bk k ⊆ setA := by
  intro y hy
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨k, Or.inl (Or.inr hy)⟩

lemma Fk_sub_setA (k : ℕ) : Fk k ⊆ setA := by
  intro y hy
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨k, Or.inr hy⟩

lemma I_sub_setA (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ setA := by
  cases k with
  | zero =>
      intro y hy
      simp only [Q, pow_zero, mul_one, mem_Icc] at hy
      obtain ⟨h1, h2⟩ := hy
      interval_cases y
      · exact two_mem
      · exact three_mem
  | succ m =>
      intro y hy
      simp only [mem_Icc] at hy
      have hQ : Q (m + 1) = 5 * Q m := Q_succ m
      have hm := Q_pos m
      apply Fk_sub_setA m
      simp only [Fk, mem_Icc]
      omega

lemma mem_I (k : ℕ) {y : ℕ} (h1 : 2 * Q k ≤ y) (h2 : y ≤ 3 * Q k) : y ∈ setA :=
  I_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_Bk (k : ℕ) {y : ℕ} (h1 : 5 * Q k ≤ y) (h2 : y ≤ 6 * Q k - 1) : y ∈ setA :=
  Bk_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_Fk (k : ℕ) {y : ℕ} (h1 : 10 * Q k - 1 ≤ y) (h2 : y ≤ 15 * Q k) : y ∈ setA :=
  Fk_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

/-! ## Classification of elements of setA -/

lemma setA_cases {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨
      (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_insert_iff,
    Set.mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
  rcases hx with (h | h) | ⟨j, (h | h) | h⟩
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr ⟨j, Or.inl h⟩)
  · exact Or.inr (Or.inr ⟨j, Or.inr (Or.inl h)⟩)
  · exact Or.inr (Or.inr ⟨j, Or.inr (Or.inr h)⟩)

lemma setA_pos {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases setA_cases hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have := Q_pos j; rcases hj with h | h | h <;> omega

/-- Every element of setA, relative to level k, falls into a "small", "stage-k", or "large" band. -/
lemma band (x k : ℕ) (hx : x ∈ setA) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨
      (10 * Q k - 1 ≤ x ∧ x ≤ 15 * Q k) ∨ 20 * Q k ≤ x := by
  have hk := Q_pos k
  rcases setA_cases hx with h2 | h3 | ⟨j, hj⟩
  · left; omega
  · left; omega
  · have hjp := Q_pos j
    rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have h5 : 5 * Q j ≤ Q k := by
        have hh : Q (j + 1) ≤ Q k := Q_mono (Nat.succ_le_of_lt hlt)
        rw [Q_succ] at hh; omega
      rcases hj with h | h | h <;> omega
    · rw [hje] at hj
      rcases hj with h | h | h
      · right; left; exact h
      · right; right; left; exact h
      · right; right; right; left; exact h
    · right; right; right; right
      have h5 : 5 * Q k ≤ Q j := by
        have hh : Q (k + 1) ≤ Q j := Q_mono (Nat.succ_le_of_lt hgt)
        rw [Q_succ] at hh; omega
      rcases hj with h | h | h <;> omega

/-! ## Basis: setA + setA covers every n ≥ 4 -/

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ setA + setA := by
  induction k with
  | zero =>
      intro x hx
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      obtain ⟨h4, h6⟩ := hx
      rw [Set.mem_add]
      interval_cases x
      · exact ⟨2, two_mem, 2, two_mem, rfl⟩
      · exact ⟨2, two_mem, 3, three_mem, rfl⟩
      · exact ⟨3, three_mem, 3, three_mem, rfl⟩
  | succ k ih =>
      intro x hx
      simp only [mem_Icc] at hx
      obtain ⟨hx4, hxhi⟩ := hx
      have hQs : Q (k + 1) = 5 * Q k := Q_succ k
      have hk := Q_pos k
      by_cases hsmall : x ≤ 6 * Q k
      · exact ih (mem_Icc.mpr ⟨hx4, hsmall⟩)
      · rw [Set.mem_add]
        by_cases hA : x ≤ 7 * Q k
        · exact ⟨4 * Q k, ck_mem k, x - 4 * Q k, mem_I k (by omega) (by omega), by omega⟩
        · by_cases hB : x ≤ 9 * Q k - 1
          · refine ⟨max (2 * Q k) (x - (6 * Q k - 1)), mem_I k (by omega) (by omega),
                   x - max (2 * Q k) (x - (6 * Q k - 1)), mem_Bk k (by omega) (by omega), by omega⟩
          · by_cases hC : x ≤ 10 * Q k - 1
            · exact ⟨4 * Q k, ck_mem k, x - 4 * Q k, mem_Bk k (by omega) (by omega), by omega⟩
            · by_cases hD : x ≤ 12 * Q k - 2
              · refine ⟨max (5 * Q k) (x - (6 * Q k - 1)), mem_Bk k (by omega) (by omega),
                       x - max (5 * Q k) (x - (6 * Q k - 1)), mem_Bk k (by omega) (by omega), by omega⟩
              · by_cases hE : x ≤ 18 * Q k
                · refine ⟨max (2 * Q k) (x - 15 * Q k), mem_I k (by omega) (by omega),
                         x - max (2 * Q k) (x - 15 * Q k), mem_Fk k (by omega) (by omega), by omega⟩
                · by_cases hF : x ≤ 21 * Q k - 1
                  · refine ⟨max (5 * Q k) (x - 15 * Q k), mem_Bk k (by omega) (by omega),
                           x - max (5 * Q k) (x - 15 * Q k), mem_Fk k (by omega) (by omega), by omega⟩
                  · refine ⟨max (10 * Q k - 1) (x - 15 * Q k), mem_Fk k (by omega) (by omega),
                           x - max (10 * Q k - 1) (x - 15 * Q k), mem_Fk k (by omega) (by omega), by omega⟩

/-! ## Rigidity and gap -/

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have ha2 := setA_pos ha
  have hb2 := setA_pos hb
  have hk := Q_pos k
  rcases band a k ha with ha' | ha' | ha' | ha' | ha' <;>
    rcases band b k hb with hb' | hb' | hb' | hb' | hb' <;>
    first
      | (exfalso; omega)
      | exact Or.inl ⟨by show a = 4 * Q k; omega,
          by show b ∈ Icc (5 * Q k) (6 * Q k - 1); rw [mem_Icc]; omega⟩
      | exact Or.inr ⟨by show b = 4 * Q k; omega,
          by show a ∈ Icc (5 * Q k) (6 * Q k - 1); rw [mem_Icc]; omega⟩

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hnJ, hnsum⟩
  rw [Set.mem_add] at hnsum
  obtain ⟨a, ha, b, hb, hab⟩ := hnsum
  have ha' := hT ha
  have hb' := hT hb
  rcases rigidity k n hnJ a b ha' hb' hab with ⟨hcka, _⟩ | ⟨hckb, _⟩
  · exact hck (hcka ▸ ha)
  · exact hck (hckb ▸ hb)

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
  · -- basis
    intro n hn
    have hkn : n ≤ 6 * Q n := by have h1 := n_lt_Q n; omega
    have hmem := basis_lem n (mem_Icc.mpr ⟨hn, hkn⟩)
    rw [Set.mem_add] at hmem
    exact hmem
  · -- no partition is both-syndetic
    intro A₁ A₂ h1 h2 hcov hdisj hboth
    obtain ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩ := hboth
    set k := max C₁ C₂ with hk_def
    have hle1 : C₁ ≤ k := by rw [hk_def]; exact le_max_left _ _
    have hle2 : C₂ ≤ k := by rw [hk_def]; exact le_max_right _ _
    have hkk : k < Q k := n_lt_Q k
    have hckmem : (4 * Q k) ∈ setA := ck_mem k
    rcases hcov _ hckmem with hin1 | hin2
    · have hnotA2 : (4 * Q k) ∉ A₂ := by
        intro hmem
        have hcon : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hin1, hmem⟩
        rw [hdisj] at hcon; exact hcon
      have hgap := gap_lem k A₂ h2 hnotA2
      obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        show m ∈ Ico (9 * Q k) (10 * Q k)
        rw [mem_Ico]; omega
      have hfin : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hfin; exact hfin
    · have hnotA1 : (4 * Q k) ∉ A₁ := by
        intro hmem
        have hcon : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hmem, hin2⟩
        rw [hdisj] at hcon; exact hcon
      have hgap := gap_lem k A₁ h1 hnotA1
      obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        show m ∈ Ico (9 * Q k) (10 * Q k)
        rw [mem_Ico]; omega
      have hfin : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hfin; exact hfin

end Erdos741OAI

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
def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | (k+1) => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic arithmetic on Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k
lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by simp [Q, pow_succ, mul_comm]

lemma k_le_Qk (k : ℕ) : k ≤ Q k := by
  induction k with
  | zero => simp [Q]
  | succ k ih =>
    have := Q_pos k
    rw [Q_succ]; omega

/-! ## Akn monotone and contained in setA -/

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  show x ∈ Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)
  exact Or.inl hx

lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, hx⟩)

/-! ## The inherited interval I = [2Qk, 3Qk] ⊆ Akn k -/

lemma I_sub_akn (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn k := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one] at hx
    exact hx
  | succ m =>
    intro x hx
    show x ∈ Akn m ∪ ({ck m} ∪ Bk m ∪ Fk m)
    refine Or.inr (Or.inr ?_)
    rw [mem_Icc] at hx
    have hQ := Q_succ m
    exact mem_Icc.mpr ⟨by omega, by omega⟩

/-! ## Basis lemma: Icc 4 (6Qk) ⊆ Akn k + Akn k -/

lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  intro k
  induction k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    simp only [Q, pow_zero, mul_one] at hx
    rw [Set.mem_add]
    by_cases h : x ≤ 5
    · exact ⟨2, mem_Icc.mpr ⟨by omega, by omega⟩, x - 2, mem_Icc.mpr ⟨by omega, by omega⟩, by omega⟩
    · exact ⟨3, mem_Icc.mpr ⟨by omega, by omega⟩, x - 3, mem_Icc.mpr ⟨by omega, by omega⟩, by omega⟩
  | succ k ih =>
    have hQ := Q_succ k
    have hQpos := Q_pos k
    have hck : (4 * Q k) ∈ Akn (k+1) := by
      show 4 * Q k ∈ Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)
      exact Or.inr (Or.inl (Or.inl rfl))
    have hI' : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k+1) := by
      intro y h1 h2
      exact akn_mono k (I_sub_akn k (mem_Icc.mpr ⟨h1, h2⟩))
    have hBk : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k+1) := by
      intro y h1 h2
      show y ∈ Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)
      exact Or.inr (Or.inl (Or.inr (mem_Icc.mpr ⟨h1, h2⟩)))
    have hFk : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k+1) := by
      intro y h1 h2
      show y ∈ Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)
      exact Or.inr (Or.inr (mem_Icc.mpr ⟨h1, h2⟩))
    intro x hx
    rw [mem_Icc] at hx
    obtain ⟨hx4, hxhi⟩ := hx
    by_cases hlow : x ≤ 6 * Q k
    · have hxin : x ∈ Icc 4 (6 * Q k) := mem_Icc.mpr ⟨hx4, hlow⟩
      exact Set.add_subset_add (akn_mono k) (akn_mono k) (ih hxin)
    · push_neg at hlow
      rw [Set.mem_add]
      by_cases c1 : x ≤ 7 * Q k
      · exact ⟨4 * Q k, hck, x - 4 * Q k, hI' _ (by omega) (by omega), by omega⟩
      · push_neg at c1
        by_cases c2 : x ≤ 9 * Q k - 1
        · refine ⟨max (2 * Q k) (x - (6 * Q k - 1)), hI' _ (by omega) (by omega),
                 x - max (2 * Q k) (x - (6 * Q k - 1)), hBk _ (by omega) (by omega), by omega⟩
        · push_neg at c2
          by_cases c3 : x ≤ 10 * Q k - 1
          · exact ⟨4 * Q k, hck, x - 4 * Q k, hBk _ (by omega) (by omega), by omega⟩
          · push_neg at c3
            by_cases c4 : x ≤ 12 * Q k - 2
            · refine ⟨max (5 * Q k) (x - (6 * Q k - 1)), hBk _ (by omega) (by omega),
                     x - max (5 * Q k) (x - (6 * Q k - 1)), hBk _ (by omega) (by omega), by omega⟩
            · push_neg at c4
              by_cases c5 : x ≤ 18 * Q k
              · refine ⟨max (2 * Q k) (x - 15 * Q k), hI' _ (by omega) (by omega),
                       x - max (2 * Q k) (x - 15 * Q k), hFk _ (by omega) (by omega), by omega⟩
              · push_neg at c5
                by_cases c6 : x ≤ 21 * Q k - 1
                · refine ⟨max (5 * Q k) (x - 15 * Q k), hBk _ (by omega) (by omega),
                         x - max (5 * Q k) (x - 15 * Q k), hFk _ (by omega) (by omega), by omega⟩
                · push_neg at c6
                  refine ⟨max (10 * Q k - 1) (x - 15 * Q k), hFk _ (by omega) (by omega),
                         x - max (10 * Q k - 1) (x - 15 * Q k), hFk _ (by omega) (by omega), by omega⟩

/-! ## setA elements are ≥ 2 -/

lemma setA_ge (x : ℕ) (hx : x ∈ setA) : 2 ≤ x := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
  rcases hx with hsmall | ⟨j, hj⟩
  · omega
  · have := Q_pos j
    rcases hj with (hc | hB) | hF <;> omega

/-! ## Classification of setA elements relative to scale k -/

lemma classify (k x : ℕ) (hx : x ∈ setA) (hb : x ≤ 10 * Q k - 3) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hx
  rcases hx with hsmall | ⟨j, hj⟩
  · have := Q_pos k; left; omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have h5 : 5 * Q j ≤ Q k := by
        have hp : (5 : ℕ) ^ (j+1) ≤ 5 ^ k := Nat.pow_le_pow_right (by norm_num) hlt
        have hh : Q (j+1) ≤ Q k := hp
        rw [Q_succ] at hh; exact hh
      have := Q_pos j
      rcases hj with (hc | hB) | hF <;> omega
    · rw [hje] at hj
      rcases hj with (hc | hB) | hF
      · right; left; omega
      · right; right; exact hB
      · exfalso; have := Q_pos k; omega
    · exfalso
      have h5 : 5 * Q k ≤ Q j := by
        have hp : (5 : ℕ) ^ (k+1) ≤ 5 ^ j := Nat.pow_le_pow_right (by norm_num) hgt
        have hh : Q (k+1) ≤ Q j := hp
        rw [Q_succ] at hh; exact hh
      have := Q_pos k
      rcases hj with (hc | hB) | hF <;> omega

/-! ## Rigidity: sums into Jk k must use the connector ck k -/

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) (a b : ℕ)
    (ha : a ∈ setA) (hbb : b ∈ setA) (hab : a + b = n) :
    (a = 4 * Q k ∧ b ∈ Bk k) ∨ (b = 4 * Q k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hge2a : 2 ≤ a := setA_ge a ha
  have hge2b : 2 ≤ b := setA_ge b hbb
  have hQ := Q_pos k
  have hale : a ≤ 10 * Q k - 3 := by omega
  have hble : b ≤ 10 * Q k - 3 := by omega
  have ca := classify k a ha hale
  have cb := classify k b hbb hble
  rcases ca with ca | ca | ca
  · rcases cb with cb | cb | cb
    · exfalso; omega
    · exfalso; omega
    · exfalso; obtain ⟨cb1, cb2⟩ := cb; omega
  · rcases cb with cb | cb | cb
    · exfalso; omega
    · exfalso; omega
    · left; obtain ⟨cb1, cb2⟩ := cb
      exact ⟨ca, mem_Icc.mpr ⟨cb1, cb2⟩⟩
  · obtain ⟨ca1, ca2⟩ := ca
    rcases cb with cb | cb | cb
    · exfalso; omega
    · right; exact ⟨cb, mem_Icc.mpr ⟨ca1, ca2⟩⟩
    · exfalso; obtain ⟨cb1, cb2⟩ := cb; omega

/-! ## Gap lemma: without the connector, Jk k is unreachable -/

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hnotin : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hnJ, hsum⟩
  rw [Set.mem_add] at hsum
  obtain ⟨a, ha, b, hb, hab⟩ := hsum
  have ha' : a ∈ setA := hT ha
  have hb' : b ∈ setA := hT hb
  rcases rigidity k n hnJ a b ha' hb' hab with ⟨h1, _⟩ | ⟨h1, _⟩
  · apply hnotin
    show (4 * Q k) ∈ T
    rw [← h1]; exact ha
  · apply hnotin
    show (4 * Q k) ∈ T
    rw [← h1]; exact hb

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
    have hmem : n ∈ Icc 4 (6 * Q n) := by
      have := k_le_Qk n; have := Q_pos n
      exact mem_Icc.mpr ⟨hn, by omega⟩
    have hcov := basis_lem n hmem
    rw [Set.mem_add] at hcov
    obtain ⟨a, ha, b, hb, hab⟩ := hcov
    exact ⟨a, akn_sub_setA n ha, b, akn_sub_setA n hb, hab⟩
  · rintro A₁ A₂ h1 h2 hcov hdisj ⟨⟨C₁, hsyn1⟩, ⟨C₂, hsyn2⟩⟩
    set k := C₁ + C₂ + 1 with hk
    have hQk1 : C₁ < Q k := by have := k_le_Qk k; omega
    have hQk2 : C₂ < Q k := by have := k_le_Qk k; omega
    have hckA : ck k ∈ setA := by
      have : ck k ∈ ⋃ j, ({ck j} ∪ Bk j ∪ Fk j) :=
        Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩
      exact Or.inr this
    rcases hcov (ck k) hckA with hin1 | hin2
    · have hnotin : ck k ∉ A₂ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hc⟩
        rw [hdisj] at this; exact this
      have hgap := gap_lem k A₂ h2 hnotin
      obtain ⟨m, hmA, hmI⟩ := hsyn2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; exact ⟨by omega, by omega⟩
      have hmem : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmA⟩
      rw [hgap] at hmem; exact hmem
    · have hnotin : ck k ∉ A₁ := by
        intro hc
        have : ck k ∈ A₁ ∩ A₂ := ⟨hc, hin2⟩
        rw [hdisj] at this; exact this
      have hgap := gap_lem k A₁ h1 hnotin
      obtain ⟨m, hmA, hmI⟩ := hsyn1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; exact ⟨by omega, by omega⟩
      have hmem : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmA⟩
      rw [hgap] at hmem; exact hmem

end Erdos741OAI

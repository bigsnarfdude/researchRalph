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
def Stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, Stage k
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k + 1) => Akn k ∪ Stage k

/-! ## Arithmetic helpers -/

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by unfold Q; rw [pow_succ]; ring

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma le_Q (k : ℕ) : k ≤ Q k := by
  induction k with
  | zero => exact Nat.zero_le _
  | succ k ih =>
    have hq := Q_pos k
    rw [Q_succ]; omega

lemma five_Q_le {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have h1 : Q (j + 1) ≤ Q k := Q_mono (by omega)
  rw [Q_succ] at h1; exact h1

/-! ## Akn is contained in setA and is monotone -/

lemma Akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := fun x hx => Or.inl hx

lemma Akn_sub_setA : ∀ k, Akn k ⊆ setA := by
  intro k
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Or.inr (mem_iUnion.mpr ⟨k, hx⟩)

/-! ## Membership classification -/

lemma setA_cases (x : ℕ) (hx : x ∈ setA) :
    (x = 2 ∨ x = 3) ∨ ∃ j, (x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1)) ∨
      (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  simp only [setA, Stage, ck, Bk, Fk, mem_union, mem_iUnion, mem_singleton_iff,
    mem_insert_iff, mem_Icc] at hx
  exact hx

lemma two_le_of_mem {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases setA_cases x hx with (h | h) | ⟨j, (h | ⟨h, _⟩) | ⟨h, _⟩⟩
  · omega
  · omega
  · have := Q_pos j; omega
  · have := Q_pos j; omega
  · have := Q_pos j; omega

lemma classify (k y : ℕ) (hy : y ∈ setA) (hlt : y < 10 * Q k) :
    y ≤ 3 * Q k ∨ y = 4 * Q k ∨ (5 * Q k ≤ y ∧ y ≤ 6 * Q k - 1) ∨ y = 10 * Q k - 1 := by
  have hqk := Q_pos k
  rcases setA_cases y hy with (h | h) | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt' | heq | hgt'
    · have h5 := five_Q_le hlt'
      have hqj := Q_pos j
      rcases hj with (h | ⟨h1, h2⟩) | ⟨h1, h2⟩
      · left; omega
      · left; omega
      · left; omega
    · rw [heq] at hj
      rcases hj with (h | ⟨h1, h2⟩) | ⟨h1, h2⟩
      · right; left; omega
      · right; right; left; exact ⟨h1, h2⟩
      · right; right; right; omega
    · have h5 := five_Q_le hgt'
      have hqj := Q_pos j
      rcases hj with (h | ⟨h1, h2⟩) | ⟨h1, h2⟩
      · exfalso; omega
      · exfalso; omega
      · exfalso; omega

/-! ## Rigidity and gap lemmas -/

lemma rigidity (k : ℕ) {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b ∈ Ico (9 * Q k) (10 * Q k)) : a = ck k ∨ b = ck k := by
  rw [mem_Ico] at hab
  obtain ⟨hlo, hhi⟩ := hab
  have ha2 := two_le_of_mem ha
  have hb2 := two_le_of_mem hb
  have hca := classify k a ha (by omega)
  have hcb := classify k b hb (by omega)
  rcases hca with hA | hA | hA | hA <;> rcases hcb with hB | hB | hB | hB <;>
    simp only [ck] <;> omega

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hc : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [eq_empty_iff_forall_not_mem]
  intro m hm
  rw [mem_inter_iff] at hm
  obtain ⟨hmJ, hmTT⟩ := hm
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmTT
  have haA := hT ha
  have hbA := hT hb
  have hmJ' : a + b ∈ Ico (9 * Q k) (10 * Q k) := by rw [hab]; exact hmJ
  rcases rigidity k haA hbA hmJ' with h | h
  · exact hc (h ▸ ha)
  · exact hc (h ▸ hb)

/-! ## Basis lemma -/

lemma base_cover : Icc 4 30 ⊆ Akn 1 + Akn 1 := by
  have hT : ∀ x, x ∈ ({2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15} : Finset ℕ) → x ∈ Akn 1 := by
    intro x hx
    fin_cases hx <;>
      simp only [Akn, Stage, ck, Bk, Fk, Q, pow_zero, pow_one, mul_one, mem_union,
        mem_singleton_iff, mem_insert_iff, mem_Icc] <;> omega
  intro n hn
  rw [mem_Icc] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hcov : ∃ a ∈ ({2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15} : Finset ℕ),
      ∃ b ∈ ({2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15} : Finset ℕ), a + b = n := by
    interval_cases n <;> decide
  obtain ⟨a, haT, b, hbT, hab⟩ := hcov
  exact Set.mem_add.mpr ⟨a, hT a haT, b, hT b hbT, hab⟩

lemma cover (k : ℕ) : Icc 4 (30 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero => simpa [Q] using base_cover
  | succ k ih =>
    intro n hn
    rw [mem_Icc] at hn
    obtain ⟨h4, hub⟩ := hn
    by_cases hsmall : n ≤ 30 * Q k
    · obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (ih (mem_Icc.mpr ⟨h4, hsmall⟩))
      exact Set.mem_add.mpr ⟨a, Akn_mono _ ha, b, Akn_mono _ hb, hab⟩
    · have hq := Q_succ k
      have hck : ck (k + 1) = 4 * Q (k + 1) := rfl
      have hpos := Q_pos (k + 1)
      have subAkn : Stage (k + 1) ⊆ Akn (k + 1 + 1) := fun x hx => Or.inr hx
      have subAkn1 : Akn (k + 1) ⊆ Akn (k + 1 + 1) := Akn_mono (k + 1)
      have hc : ck (k + 1) ∈ Akn (k + 1 + 1) := subAkn (Or.inl (Or.inl rfl))
      have hB : ∀ x, 5 * Q (k + 1) ≤ x → x ≤ 6 * Q (k + 1) - 1 → x ∈ Akn (k + 1 + 1) := by
        intro x h1 h2
        exact subAkn (Or.inl (Or.inr (mem_Icc.mpr ⟨h1, h2⟩)))
      have hF : ∀ x, 10 * Q (k + 1) - 1 ≤ x → x ≤ 15 * Q (k + 1) → x ∈ Akn (k + 1 + 1) := by
        intro x h1 h2
        exact subAkn (Or.inr (mem_Icc.mpr ⟨h1, h2⟩))
      have hI : ∀ x, 2 * Q (k + 1) ≤ x → x ≤ 3 * Q (k + 1) → x ∈ Akn (k + 1 + 1) := by
        intro x h1 h2
        apply subAkn1
        exact Or.inr (Or.inr (mem_Icc.mpr ⟨by omega, by omega⟩))
      by_cases c1 : n ≤ 7 * Q (k + 1)
      · exact Set.mem_add.mpr ⟨n - 4 * Q (k + 1), hI _ (by omega) (by omega),
          ck (k + 1), hc, by omega⟩
      by_cases c2 : n ≤ 8 * Q (k + 1)
      · exact Set.mem_add.mpr ⟨n - 5 * Q (k + 1), hI _ (by omega) (by omega),
          5 * Q (k + 1), hB _ (by omega) (by omega), by omega⟩
      by_cases c3 : n ≤ 9 * Q (k + 1) - 1
      · exact Set.mem_add.mpr ⟨n - (6 * Q (k + 1) - 1), hI _ (by omega) (by omega),
          6 * Q (k + 1) - 1, hB _ (by omega) (by omega), by omega⟩
      by_cases c4 : n ≤ 10 * Q (k + 1) - 1
      · exact Set.mem_add.mpr ⟨n - 4 * Q (k + 1), hB _ (by omega) (by omega),
          ck (k + 1), hc, by omega⟩
      by_cases c5 : n ≤ 11 * Q (k + 1) - 1
      · exact Set.mem_add.mpr ⟨n - 5 * Q (k + 1), hB _ (by omega) (by omega),
          5 * Q (k + 1), hB _ (by omega) (by omega), by omega⟩
      by_cases c6 : n ≤ 12 * Q (k + 1) - 2
      · exact Set.mem_add.mpr ⟨n - (6 * Q (k + 1) - 1), hB _ (by omega) (by omega),
          6 * Q (k + 1) - 1, hB _ (by omega) (by omega), by omega⟩
      by_cases c7 : n ≤ 17 * Q (k + 1)
      · exact Set.mem_add.mpr ⟨2 * Q (k + 1), hI _ (by omega) (by omega),
          n - 2 * Q (k + 1), hF _ (by omega) (by omega), by omega⟩
      by_cases c8 : n ≤ 18 * Q (k + 1)
      · exact Set.mem_add.mpr ⟨3 * Q (k + 1), hI _ (by omega) (by omega),
          n - 3 * Q (k + 1), hF _ (by omega) (by omega), by omega⟩
      by_cases c9 : n ≤ 20 * Q (k + 1)
      · exact Set.mem_add.mpr ⟨5 * Q (k + 1), hB _ (by omega) (by omega),
          n - 5 * Q (k + 1), hF _ (by omega) (by omega), by omega⟩
      by_cases c10 : n ≤ 21 * Q (k + 1) - 1
      · exact Set.mem_add.mpr ⟨6 * Q (k + 1) - 1, hB _ (by omega) (by omega),
          n - (6 * Q (k + 1) - 1), hF _ (by omega) (by omega), by omega⟩
      by_cases c11 : n ≤ 25 * Q (k + 1) - 1
      · exact Set.mem_add.mpr ⟨10 * Q (k + 1) - 1, hF _ (by omega) (by omega),
          n - (10 * Q (k + 1) - 1), hF _ (by omega) (by omega), by omega⟩
      · exact Set.mem_add.mpr ⟨15 * Q (k + 1), hF _ (by omega) (by omega),
          n - 15 * Q (k + 1), hF _ (by omega) (by omega), by omega⟩

lemma basis_lem (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have hk : n ≤ 30 * Q n := by have := le_Q n; have := Q_pos n; omega
  have hmem : n ∈ Icc 4 (30 * Q n) := mem_Icc.mpr ⟨hn, hk⟩
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp (cover n hmem)
  exact ⟨a, Akn_sub_setA _ ha, b, Akn_sub_setA _ hb, hab⟩

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
    exact basis_lem n hn
  · intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    obtain ⟨k, hQC1, hQC2⟩ : ∃ k, C₁ < Q k ∧ C₂ < Q k := by
      refine ⟨C₁ + C₂ + 1, ?_, ?_⟩
      · have := le_Q (C₁ + C₂ + 1); omega
      · have := le_Q (C₁ + C₂ + 1); omega
    have hckA : ck k ∈ setA := by
      have hs : ck k ∈ Stage k := by simp [Stage]
      exact Or.inr (mem_iUnion.mpr ⟨k, hs⟩)
    rcases hcov (ck k) hckA with hin1 | hin2
    · have hnotA2 : ck k ∉ A₂ := by
        intro hmem
        have hx : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hmem⟩
        rw [hdisj] at hx; exact hx
      have hgap := gap_lem k A₂ h2 hnotA2
      obtain ⟨m, hmTT, hmIcc⟩ := hC2 (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmTT⟩
      rw [hgap] at hcontra; exact hcontra
    · have hnotA1 : ck k ∉ A₁ := by
        intro hmem
        have hx : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hin2⟩
        rw [hdisj] at hx; exact hx
      have hgap := gap_lem k A₁ h1 hnotA1
      obtain ⟨m, hmTT, hmIcc⟩ := hC1 (9 * Q k)
      rw [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmTT⟩
      rw [hgap] at hcontra; exact hcontra

end Erdos741OAI

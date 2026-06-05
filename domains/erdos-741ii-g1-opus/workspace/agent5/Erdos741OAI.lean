import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

-- YOUR TASK: implement the construction described in program.md and prove the theorem below.

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## The construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma one_le_Q (k : ℕ) : 1 ≤ Q k := Q_pos k

/-! ### Membership helpers for Akn -/

lemma akn_subset_succ (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inl (Or.inl (Or.inl hx))

lemma ck_mem_akn_succ (k : ℕ) : ck k ∈ Akn (k+1) := by
  simp only [Akn, mem_union, mem_singleton_iff]
  tauto

lemma Bk_subset_akn_succ (k : ℕ) : Bk k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inl (Or.inr hx)

lemma Fk_subset_akn_succ (k : ℕ) : Fk k ⊆ Akn (k+1) := by
  intro x hx
  simp only [Akn, mem_union]
  exact Or.inr hx

-- monotonicity: Akn k ⊆ Akn (k+j)
lemma akn_mono {k j : ℕ} (h : k ≤ j) : Akn k ⊆ Akn j := by
  induction h with
  | refl => exact subset_rfl
  | @step m _ ih => exact ih.trans (akn_subset_succ m)

lemma two_mem_akn (k : ℕ) : 2 ∈ Akn k := by
  have h0 : (2:ℕ) ∈ Akn 0 := by simp [Akn]
  exact akn_mono (Nat.zero_le k) h0

lemma three_mem_akn (k : ℕ) : 3 ∈ Akn k := by
  have h0 : (3:ℕ) ∈ Akn 0 := by simp [Akn]
  exact akn_mono (Nat.zero_le k) h0

-- Fk at level k lives inside Akn (k+1), hence inside Akn (k+2) etc.
lemma Fk_subset_akn {k j : ℕ} (h : k + 1 ≤ j) : Fk k ⊆ Akn j :=
  (Fk_subset_akn_succ k).trans (akn_mono h)

lemma Bk_subset_akn {k j : ℕ} (h : k + 1 ≤ j) : Bk k ⊆ Akn j :=
  (Bk_subset_akn_succ k).trans (akn_mono h)

lemma ck_mem_akn {k j : ℕ} (h : k + 1 ≤ j) : ck k ∈ Akn j :=
  akn_mono h (ck_mem_akn_succ k)

/-! ### setA containment -/

lemma stage_subset_setA (k : ℕ) : stage k ⊆ setA := by
  intro x hx
  exact Or.inr (mem_iUnion.mpr ⟨k, hx⟩)

lemma akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    simp only [Akn, mem_union] at hx
    rcases hx with (((h | h) | h) | h)
    · exact ih h
    · rw [mem_singleton_iff] at h; subst h
      exact stage_subset_setA k (Or.inl (Or.inl (mem_singleton _)))
    · exact stage_subset_setA k (Or.inl (Or.inr h))
    · exact stage_subset_setA k (Or.inr h)

/-! ### Basis: A covers all n ≥ 4 -/

-- the sum of two integer intervals is a full interval
lemma sum_interval {a0 a1 b0 b1 x : ℕ} (ha : a0 ≤ a1) (hb : b0 ≤ b1)
    (hx0 : a0 + b0 ≤ x) (hx1 : x ≤ a1 + b1) :
    ∃ a, (a0 ≤ a ∧ a ≤ a1) ∧ ∃ b, (b0 ≤ b ∧ b ≤ b1) ∧ a + b = x := by
  by_cases hc : x ≤ a1 + b0
  · exact ⟨x - b0, by omega, b0, by omega, by omega⟩
  · exact ⟨a1, by omega, x - a1, by omega, by omega⟩

lemma mem_Fk_akn {k j a : ℕ} (hj : k + 1 ≤ j)
    (h : 10 * Q k - 1 ≤ a ∧ a ≤ 15 * Q k) : a ∈ Akn j :=
  Fk_subset_akn hj (by rw [Fk]; exact mem_Icc.mpr h)

lemma mem_Bk_akn {k j a : ℕ} (hj : k + 1 ≤ j)
    (h : 5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1) : a ∈ Akn j :=
  Bk_subset_akn hj (by rw [Bk]; exact mem_Icc.mpr h)

-- membership in Akn 1 for the base case
lemma in_akn1 (x : ℕ) (h : (2 ≤ x ∧ x ≤ 5) ∨ (9 ≤ x ∧ x ≤ 15)) : x ∈ Akn 1 := by
  simp only [Akn, Q, ck, Bk, Fk, mem_union, mem_insert_iff, mem_singleton_iff, mem_Icc,
    pow_zero, mul_one]
  omega

-- Q n grows past n
lemma n_le_Q_succ (n : ℕ) : n ≤ 6 * Q (n + 1) := by
  have h5 : n < 5 ^ n := by
    calc n < 2 ^ n := Nat.lt_two_pow_self
    _ ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  have hmono : Q n ≤ Q (n + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hQn : Q n = 5 ^ n := rfl
  omega

lemma basis_cover : ∀ k, Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [Q, mem_Icc] at hx
    -- hx : 4 ≤ x ∧ x ≤ 30
    rw [Set.mem_add]
    obtain ⟨hlo, hhi⟩ := hx
    by_cases h1 : x ≤ 7
    · exact ⟨2, in_akn1 2 (by omega), x - 2, in_akn1 (x-2) (by omega), by omega⟩
    by_cases h2 : x ≤ 10
    · exact ⟨5, in_akn1 5 (by omega), x - 5, in_akn1 (x-5) (by omega), by omega⟩
    by_cases h3 : x ≤ 17
    · exact ⟨2, in_akn1 2 (by omega), x - 2, in_akn1 (x-2) (by omega), by omega⟩
    by_cases h4 : x ≤ 24
    · exact ⟨9, in_akn1 9 (by omega), x - 9, in_akn1 (x-9) (by omega), by omega⟩
    · exact ⟨15, in_akn1 15 (by omega), x - 15, in_akn1 (x-15) (by omega), by omega⟩
  | succ k ih =>
    intro x hx
    simp only [mem_Icc] at hx
    obtain ⟨hx4, hxhi⟩ := hx
    have hQ1 : Q (k+1) = 5 * Q k := Q_succ k
    have hQ2 : Q (k+1+1) = 5 * Q (k+1) := Q_succ (k+1)
    have hpos : 0 < Q k := Q_pos k
    have hck1 : ck (k+1) = 4 * Q (k+1) := rfl
    rw [Set.mem_add]
    by_cases hA : x ≤ 6 * Q (k+1)
    · -- low part: inherit from IH and lift
      have hin : x ∈ Akn (k+1) + Akn (k+1) := ih (mem_Icc.mpr ⟨hx4, hA⟩)
      rw [Set.mem_add] at hin
      obtain ⟨a, ha, b, hb, hs⟩ := hin
      exact ⟨a, akn_subset_succ (k+1) ha, b, akn_subset_succ (k+1) hb, hs⟩
    by_cases hB : x ≤ 7 * Q (k+1)
    · -- I + ck
      exact ⟨x - ck (k+1), mem_Fk_akn (k := k) (by omega) (by omega),
             ck (k+1), ck_mem_akn_succ (k+1), by omega⟩
    by_cases hC : x ≤ 9 * Q (k+1) - 1
    · -- I + Bk
      obtain ⟨a, ⟨ha0, ha1⟩, b, ⟨hb0, hb1⟩, hs⟩ :=
        sum_interval (a0 := 10 * Q k - 1) (a1 := 15 * Q k)
          (b0 := 5 * Q (k+1)) (b1 := 6 * Q (k+1) - 1) (x := x)
          (by omega) (by omega) (by omega) (by omega)
      exact ⟨a, mem_Fk_akn (k := k) (by omega) ⟨ha0, ha1⟩,
             b, mem_Bk_akn (k := k+1) (by omega) ⟨hb0, hb1⟩, hs⟩
    by_cases hD : x ≤ 10 * Q (k+1) - 1
    · -- ck + Bk
      exact ⟨ck (k+1), ck_mem_akn_succ (k+1),
             x - ck (k+1), mem_Bk_akn (k := k+1) (by omega) (by omega), by omega⟩
    by_cases hE : x ≤ 12 * Q (k+1) - 2
    · -- Bk + Bk
      obtain ⟨a, ⟨ha0, ha1⟩, b, ⟨hb0, hb1⟩, hs⟩ :=
        sum_interval (a0 := 5 * Q (k+1)) (a1 := 6 * Q (k+1) - 1)
          (b0 := 5 * Q (k+1)) (b1 := 6 * Q (k+1) - 1) (x := x)
          (by omega) (by omega) (by omega) (by omega)
      exact ⟨a, mem_Bk_akn (k := k+1) (by omega) ⟨ha0, ha1⟩,
             b, mem_Bk_akn (k := k+1) (by omega) ⟨hb0, hb1⟩, hs⟩
    by_cases hF : x ≤ 18 * Q (k+1)
    · -- I + Fk
      obtain ⟨a, ⟨ha0, ha1⟩, b, ⟨hb0, hb1⟩, hs⟩ :=
        sum_interval (a0 := 10 * Q k - 1) (a1 := 15 * Q k)
          (b0 := 10 * Q (k+1) - 1) (b1 := 15 * Q (k+1)) (x := x)
          (by omega) (by omega) (by omega) (by omega)
      exact ⟨a, mem_Fk_akn (k := k) (by omega) ⟨ha0, ha1⟩,
             b, mem_Fk_akn (k := k+1) (by omega) ⟨hb0, hb1⟩, hs⟩
    by_cases hG : x ≤ 21 * Q (k+1) - 1
    · -- Bk + Fk
      obtain ⟨a, ⟨ha0, ha1⟩, b, ⟨hb0, hb1⟩, hs⟩ :=
        sum_interval (a0 := 5 * Q (k+1)) (a1 := 6 * Q (k+1) - 1)
          (b0 := 10 * Q (k+1) - 1) (b1 := 15 * Q (k+1)) (x := x)
          (by omega) (by omega) (by omega) (by omega)
      exact ⟨a, mem_Bk_akn (k := k+1) (by omega) ⟨ha0, ha1⟩,
             b, mem_Fk_akn (k := k+1) (by omega) ⟨hb0, hb1⟩, hs⟩
    · -- Fk + Fk
      obtain ⟨a, ⟨ha0, ha1⟩, b, ⟨hb0, hb1⟩, hs⟩ :=
        sum_interval (a0 := 10 * Q (k+1) - 1) (a1 := 15 * Q (k+1))
          (b0 := 10 * Q (k+1) - 1) (b1 := 15 * Q (k+1)) (x := x)
          (by omega) (by omega) (by omega) (by omega)
      exact ⟨a, mem_Fk_akn (k := k+1) (by omega) ⟨ha0, ha1⟩,
             b, mem_Fk_akn (k := k+1) (by omega) ⟨hb0, hb1⟩, hs⟩

lemma basis_final : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  have hmem : n ∈ Icc 4 (6 * Q (n + 1)) := mem_Icc.mpr ⟨hn, n_le_Q_succ n⟩
  have hsum := basis_cover n hmem
  rw [Set.mem_add] at hsum
  obtain ⟨a, ha, b, hb, hab⟩ := hsum
  exact ⟨a, akn_subset_setA _ ha, b, akn_subset_setA _ hb, hab⟩

/-! ### Rigidity: only ck k + Bk k sums land in Jk k -/

lemma stage_elt_bounds {j a : ℕ} (h : a ∈ stage j) : 4 * Q j ≤ a ∧ a ≤ 15 * Q j := by
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  have : 0 < Q j := Q_pos j
  omega

lemma stage_pieces {k b : ℕ} (h : b ∈ stage k) :
    b = 4 * Q k ∨ (5 * Q k ≤ b ∧ b ≤ 6 * Q k - 1) ∨ (10 * Q k - 1 ≤ b ∧ b ≤ 15 * Q k) := by
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at h
  omega

lemma setA_ge_two {a : ℕ} (ha : a ∈ setA) : 2 ≤ a := by
  simp only [setA, mem_union] at ha
  rcases ha with h | hU
  · simp only [mem_insert_iff, mem_singleton_iff] at h; omega
  · rw [mem_iUnion] at hU
    obtain ⟨j, hj⟩ := hU
    have h1 := (stage_elt_bounds hj).1
    have := Q_pos j
    omega

lemma setA_tri (k : ℕ) {a : ℕ} (ha : a ∈ setA) :
    a ≤ 3 * Q k ∨ a ∈ stage k ∨ 20 * Q k ≤ a := by
  have hpos := Q_pos k
  simp only [setA, mem_union] at ha
  rcases ha with h | hU
  · left
    simp only [mem_insert_iff, mem_singleton_iff] at h
    omega
  · rw [mem_iUnion] at hU
    obtain ⟨j, hj⟩ := hU
    rcases lt_trichotomy j k with hlt | hje | hgt
    · left
      have hb := (stage_elt_bounds hj).2
      have h1 : Q (j+1) ≤ Q k := Nat.pow_le_pow_right (by norm_num) (by omega)
      have h2 : Q (j+1) = 5 * Q j := Q_succ j
      omega
    · rw [hje] at hj; exact Or.inr (Or.inl hj)
    · right; right
      have hb := (stage_elt_bounds hj).1
      have h1 : Q (k+1) ≤ Q j := Nat.pow_le_pow_right (by norm_num) (by omega)
      have h2 : Q (k+1) = 5 * Q k := Q_succ k
      omega

lemma rigidity (k n a b : ℕ) (hn0 : 9 * Q k ≤ n) (hn1 : n < 10 * Q k)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hpos := Q_pos k
  rcases setA_tri k ha with hsa | hsa | hsa
  · rcases setA_tri k hb with hsb | hsb | hsb
    · exfalso; omega
    · exfalso
      have hbp := stage_pieces hsb
      have ha2 := setA_ge_two ha
      omega
    · exfalso; omega
  · rcases setA_tri k hb with hsb | hsb | hsb
    · exfalso
      have hap := stage_pieces hsa
      have hb2 := setA_ge_two hb
      omega
    · -- both in stage k : the rigid case
      have hap := stage_pieces hsa
      have hbp := stage_pieces hsb
      have ha4 := (stage_elt_bounds hsa).1
      have hb4 := (stage_elt_bounds hsb).1
      rcases hap with ha' | ha' | ha'
      · rcases hbp with hb' | hb' | hb'
        · exfalso; omega
        · left
          refine ⟨by rw [ck]; omega, ?_⟩
          rw [Bk]; exact mem_Icc.mpr ⟨by omega, by omega⟩
        · exfalso; omega
      · rcases hbp with hb' | hb' | hb'
        · right
          refine ⟨by rw [ck]; omega, ?_⟩
          rw [Bk]; exact mem_Icc.mpr ⟨by omega, by omega⟩
        · exfalso; omega
        · exfalso; omega
      · exfalso; omega
    · exfalso; omega
  · exfalso; omega

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n, n ∈ Jk k → n ∉ T + T := by
  intro n hn hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  simp only [Jk, mem_Ico] at hn
  rcases rigidity k n a b hn.1 hn.2 (hT ha) (hT hb) hab with ⟨hca, _⟩ | ⟨hcb, _⟩
  · exact hck (hca ▸ ha)
  · exact hck (hcb ▸ hb)

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  stage_subset_setA k (Or.inl (Or.inl (mem_singleton _)))

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, basis_final, ?_⟩
  intro A₁ A₂ h1 h2 hcov hdisj hsyn
  obtain ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩ := hsyn
  set k := C₁ + C₂ + 1 with hkdef
  have hklt : k < Q k := by
    have h5 : k < 5 ^ k := by
      calc k < 2 ^ k := Nat.lt_two_pow_self
      _ ≤ 5 ^ k := Nat.pow_le_pow_left (by norm_num) k
    have : Q k = 5 ^ k := rfl
    omega
  have hkpos := Q_pos k
  have hC1k : C₁ < Q k := by omega
  have hC2k : C₂ < Q k := by omega
  have hckA : ck k ∈ setA := ck_mem_setA k
  rcases hcov (ck k) hckA with hin1 | hin2
  · -- ck k ∈ A₁, so ck k ∉ A₂
    have hne : ck k ∉ A₂ := by
      intro hm
      have hcc : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hm⟩
      rw [hdisj] at hcc; simp at hcc
    obtain ⟨m, hmA2, hmIcc⟩ := hC2 (9 * Q k)
    simp only [mem_Icc] at hmIcc
    have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
    exact gap_lem k A₂ h2 hne m hmJ hmA2
  · -- ck k ∈ A₂, so ck k ∉ A₁
    have hne : ck k ∉ A₁ := by
      intro hm
      have hcc : ck k ∈ A₁ ∩ A₂ := ⟨hm, hin2⟩
      rw [hdisj] at hcc; simp at hcc
    obtain ⟨m, hmA1, hmIcc⟩ := hC1 (9 * Q k)
    simp only [mem_Icc] at hmIcc
    have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
    exact gap_lem k A₁ h1 hne m hmJ hmA1

end Erdos741OAI

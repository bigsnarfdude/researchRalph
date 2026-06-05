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

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ]; ring

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

lemma Q_gt_self : ∀ k, k < Q k
  | 0 => by norm_num [Q]
  | (k + 1) => by
      have ih := Q_gt_self k
      have hp := Q_pos k
      rw [Q_succ]; omega

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k + 1) => Akn k ∪ stage k

/-! ## Membership helpers -/

lemma ck_mem (k : ℕ) : (4 * Q k) ∈ Akn (k + 1) := by
  show (4 * Q k) ∈ Akn k ∪ stage k
  exact Or.inr (Or.inl (Or.inl rfl))

lemma Bk_mem {k v : ℕ} (h : v ∈ Icc (5 * Q k) (6 * Q k - 1)) : v ∈ Akn (k + 1) := by
  show v ∈ Akn k ∪ stage k
  exact Or.inr (Or.inl (Or.inr h))

lemma Fk_mem {k v : ℕ} (h : v ∈ Icc (10 * Q k - 1) (15 * Q k)) : v ∈ Akn (k + 1) := by
  show v ∈ Akn k ∪ stage k
  exact Or.inr (Or.inr h)

lemma Akn_subset_succ (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  show x ∈ Akn k ∪ stage k
  exact Or.inl hx

lemma Akn_mono {a b : ℕ} (h : a ≤ b) : Akn a ⊆ Akn b := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans (Akn_subset_succ _)

lemma Akn_sub_setA : ∀ k, Akn k ⊆ setA
  | 0 => by
      intro x hx
      exact Or.inl hx
  | (k + 1) => by
      intro x hx
      rcases (show x ∈ Akn k ∪ stage k from hx) with h | h
      · exact Akn_sub_setA k h
      · exact Or.inr (mem_iUnion.mpr ⟨k, h⟩)

-- The "I" interval at level k is inside Akn (k+1):
-- for k = 0 it is {2,3}; for k+1 it is inherited from Fk k.
lemma I_sub : ∀ k, Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1)
  | 0 => by
      intro x hx
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      show x ∈ Akn 0 ∪ stage 0
      refine Or.inl ?_
      show x ∈ ({2, 3} : Set ℕ)
      simp only [mem_insert_iff, mem_singleton_iff]
      omega
  | (k + 1) => by
      intro x hx
      simp only [mem_Icc] at hx
      have hQ := Q_succ k
      have hq := Q_pos k
      have hxF : x ∈ Icc (10 * Q k - 1) (15 * Q k) := by
        simp only [mem_Icc]; rw [hQ] at hx; omega
      show x ∈ Akn (k + 1) ∪ stage (k + 1)
      exact Or.inl (Fk_mem hxF)

lemma I_mem {k v : ℕ} (h : v ∈ Icc (2 * Q k) (3 * Q k)) : v ∈ Akn (k + 1) := I_sub k h

/-! ## Basis -/

-- The 8 pair types cover [4 Qk, 30 Qk] = [4 Qk, 6 Q(k+1)].
lemma level_cover (k : ℕ) : Icc (4 * Q k) (30 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  have hq := Q_pos k
  by_cases h1 : x ≤ 5 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, I_mem (by simp only [mem_Icc]; omega),
      x - 2 * Q k, I_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h2 : x ≤ 6 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, I_mem (by simp only [mem_Icc]; omega),
      x - 3 * Q k, I_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h3 : x ≤ 7 * Q k
  · exact Set.mem_add.mpr ⟨4 * Q k, ck_mem k,
      x - 4 * Q k, I_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h4 : x ≤ 8 * Q k
  · exact Set.mem_add.mpr ⟨5 * Q k, Bk_mem (by simp only [mem_Icc]; omega),
      x - 5 * Q k, I_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h5 : x ≤ 9 * Q k - 1
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, Bk_mem (by simp only [mem_Icc]; omega),
      x - (6 * Q k - 1), I_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h6 : x ≤ 10 * Q k - 1
  · exact Set.mem_add.mpr ⟨4 * Q k, ck_mem k,
      x - 4 * Q k, Bk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h7 : x ≤ 11 * Q k - 1
  · exact Set.mem_add.mpr ⟨5 * Q k, Bk_mem (by simp only [mem_Icc]; omega),
      x - 5 * Q k, Bk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h8 : x ≤ 12 * Q k - 2
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, Bk_mem (by simp only [mem_Icc]; omega),
      x - (6 * Q k - 1), Bk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h9 : x ≤ 17 * Q k
  · exact Set.mem_add.mpr ⟨2 * Q k, I_mem (by simp only [mem_Icc]; omega),
      x - 2 * Q k, Fk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h10 : x ≤ 18 * Q k
  · exact Set.mem_add.mpr ⟨3 * Q k, I_mem (by simp only [mem_Icc]; omega),
      x - 3 * Q k, Fk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h11 : x ≤ 20 * Q k
  · exact Set.mem_add.mpr ⟨5 * Q k, Bk_mem (by simp only [mem_Icc]; omega),
      x - 5 * Q k, Fk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h12 : x ≤ 21 * Q k - 1
  · exact Set.mem_add.mpr ⟨6 * Q k - 1, Bk_mem (by simp only [mem_Icc]; omega),
      x - (6 * Q k - 1), Fk_mem (by simp only [mem_Icc]; omega), by omega⟩
  by_cases h13 : x ≤ 25 * Q k - 1
  · exact Set.mem_add.mpr ⟨10 * Q k - 1, Fk_mem (by simp only [mem_Icc]; omega),
      x - (10 * Q k - 1), Fk_mem (by simp only [mem_Icc]; omega), by omega⟩
  · exact Set.mem_add.mpr ⟨15 * Q k, Fk_mem (by simp only [mem_Icc]; omega),
      x - 15 * Q k, Fk_mem (by simp only [mem_Icc]; omega), by omega⟩

lemma basis_cover : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1)
  | 0 => by
      intro x hx
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      apply level_cover 0
      simp only [Q, pow_zero, mul_one, mem_Icc]
      omega
  | (k + 1) => by
      intro x hx
      simp only [mem_Icc] at hx
      rw [Q_succ k] at hx
      have hq := Q_pos k
      have hmono : Akn (k + 1) + Akn (k + 1) ⊆ Akn (k + 2) + Akn (k + 2) :=
        Set.add_subset_add (Akn_mono (by omega)) (Akn_mono (by omega))
      by_cases hle : x ≤ 6 * Q k
      · exact hmono (basis_cover k (by simp only [mem_Icc]; omega))
      · exact hmono (level_cover k (by simp only [mem_Icc]; omega))

/-! ## Classification + bounds -/

lemma setA_cases {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨
      ∃ j, x = 4 * Q j ∨ x ∈ Icc (5 * Q j) (6 * Q j - 1) ∨ x ∈ Icc (10 * Q j - 1) (15 * Q j) := by
  simp only [setA, mem_union, mem_iUnion, stage, ck, Bk, Fk, mem_insert_iff,
    mem_singleton_iff] at hx
  rcases hx with (h2 | h3) | ⟨j, hj⟩
  · exact Or.inl h2
  · exact Or.inr (Or.inl h3)
  · refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (hc | hb) | hf
    · exact Or.inl hc
    · exact Or.inr (Or.inl hb)
    · exact Or.inr (Or.inr hf)

lemma two_le_mem {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases setA_cases hx with h | h | ⟨j, h | h | h⟩
  · omega
  · omega
  · have hj := Q_pos j; rw [h]; omega
  · simp only [mem_Icc] at h; have hj := Q_pos j; omega
  · simp only [mem_Icc] at h; have hj := Q_pos j; omega

lemma five_lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have := Q_mono (Nat.succ_le_of_lt h)
  rw [Q_succ] at this; exact this

lemma A_gap {x k : ℕ} (hx : x ∈ setA) (hub : x ≤ 10 * Q k - 2) : x ≤ 6 * Q k := by
  have hqk := Q_pos k
  rcases setA_cases hx with h | h | ⟨j, h | h | h⟩
  · omega
  · omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; rw [h]; omega
    · rw [hje] at h; rw [h]; omega
    · have h5 := five_lt hgt; rw [h] at hub; omega
  · simp only [mem_Icc] at h
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; omega
    · rw [hje] at h; omega
    · have h5 := five_lt hgt; omega
  · simp only [mem_Icc] at h
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; omega
    · rw [hje] at h; omega
    · have h5 := five_lt hgt; omega

lemma small_or_mid {x k : ℕ} (hx : x ∈ setA) (hub : x ≤ 6 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ x ∈ Icc (5 * Q k) (6 * Q k - 1) := by
  have hqk := Q_pos k
  rcases setA_cases hx with h | h | ⟨j, h | h | h⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; left; rw [h]; omega
    · rw [hje] at h; exact Or.inr (Or.inl h)
    · have h5 := five_lt hgt; exfalso; rw [h] at hub; omega
  · simp only [mem_Icc] at h
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; left; omega
    · rw [hje] at h; exact Or.inr (Or.inr (mem_Icc.mpr h))
    · have h5 := five_lt hgt; exfalso; omega
  · simp only [mem_Icc] at h
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 := five_lt hlt; left; omega
    · rw [hje] at h; exfalso; omega
    · have h5 := five_lt hgt; exfalso; omega

/-! ## Rigidity + gap -/

lemma rigidity {k n a b : ℕ} (hn : n ∈ Jk k) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hqk := Q_pos k
  obtain ⟨hn1, hn2⟩ := mem_Ico.mp hn
  have hb2 := two_le_mem hb
  have ha2 := two_le_mem ha
  have ha6 : a ≤ 6 * Q k := A_gap ha (by omega)
  have hb6 : b ≤ 6 * Q k := A_gap hb (by omega)
  have hca := small_or_mid ha ha6
  have hcb := small_or_mid hb hb6
  rcases hca with hA | hA | hA
  · rcases hcb with hB | hB | hB
    · exfalso; omega
    · exfalso; omega
    · exfalso; simp only [mem_Icc] at hB; omega
  · rcases hcb with hB | hB | hB
    · exfalso; omega
    · exfalso; omega
    · exact Or.inl ⟨hA, hB⟩
  · rcases hcb with hB | hB | hB
    · exfalso; simp only [mem_Icc] at hA; omega
    · exact Or.inr ⟨hB, hA⟩
    · exfalso; simp only [mem_Icc] at hA hB; omega

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hnJ, hsum⟩
  rw [Set.mem_add] at hsum
  obtain ⟨a, ha, b, hb, hab⟩ := hsum
  have := rigidity hnJ (hT ha) (hT hb) hab
  rcases this with ⟨hac, _⟩ | ⟨hbc, _⟩
  · rw [hac] at ha; exact hck ha
  · rw [hbc] at hb; exact hck hb

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
    have hcov : n ∈ Akn (n + 1) + Akn (n + 1) :=
      basis_cover n (by simp only [mem_Icc]; have := Q_gt_self n; omega)
    rw [Set.mem_add] at hcov
    obtain ⟨a, ha, b, hb, hab⟩ := hcov
    exact ⟨a, Akn_sub_setA _ ha, b, Akn_sub_setA _ hb, hab⟩
  · intro A₁ A₂ hA₁sub hA₂sub hcov hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    set k := C₁ + C₂ + 1 with hk
    have hklt := Q_gt_self k
    have hQ1 : C₁ < Q k := by omega
    have hQ2 : C₂ < Q k := by omega
    have hside := hcov (ck k) (ck_mem_setA k)
    rcases hside with hin1 | hin2
    · have hck2 : ck k ∉ A₂ := by
        intro hh
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hh⟩
        rw [hdisj] at hmem
        simp only [mem_empty_iff_false] at hmem
      have hgap := gap_lem hA₂sub hck2
      obtain ⟨m, hmS, hmI⟩ := hC₂ (9 * Q k)
      have hmJ : m ∈ Jk k := by
        simp only [mem_Icc] at hmI
        simp only [Jk, mem_Ico]
        omega
      have hmem : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hmem
      simp only [mem_empty_iff_false] at hmem
    · have hck1 : ck k ∉ A₁ := by
        intro hh
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hh, hin2⟩
        rw [hdisj] at hmem
        simp only [mem_empty_iff_false] at hmem
      have hgap := gap_lem hA₁sub hck1
      obtain ⟨m, hmS, hmI⟩ := hC₁ (9 * Q k)
      have hmJ : m ∈ Jk k := by
        simp only [mem_Icc] at hmI
        simp only [Jk, mem_Ico]
        omega
      have hmem : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hmem
      simp only [mem_empty_iff_false] at hmem

end Erdos741OAI

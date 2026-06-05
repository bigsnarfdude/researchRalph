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

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k + 1) => Akn k ∪ stage k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_le {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  simp only [Q]; exact Nat.pow_le_pow_right (by norm_num) h

lemma Q_lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have : Q (j + 1) ≤ Q k := Q_le h
  rwa [Q_succ] at this

lemma le_Q (n : ℕ) : n ≤ Q n := by
  simp only [Q]
  exact (Nat.lt_pow_self (by norm_num)).le

lemma stage_sub_setA (k : ℕ) : stage k ⊆ setA :=
  fun _ hx => Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, hx⟩)

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  stage_sub_setA k (Set.mem_union_left _ (Set.mem_union_left _ rfl))

lemma Bk_sub_setA (k : ℕ) : Bk k ⊆ setA :=
  fun _ hx => stage_sub_setA k (Set.mem_union_left _ (Set.mem_union_right _ hx))

lemma Fk_sub_setA (k : ℕ) : Fk k ⊆ setA :=
  fun _ hx => stage_sub_setA k (Set.mem_union_right _ hx)

lemma Akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  show Akn k ⊆ Akn k ∪ stage k
  exact Set.subset_union_left

lemma Akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Set.mem_union_left _ hx
  | succ k ih =>
    show Akn k ∪ stage k ⊆ setA
    exact Set.union_subset ih (stage_sub_setA k)

lemma mem_setA {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, x = ck j ∨ x ∈ Bk j ∨ x ∈ Fk j := by
  simp only [setA, stage, mem_union, mem_iUnion, mem_singleton_iff,
    Set.mem_insert_iff] at hx
  rcases hx with (h2 | h3) | ⟨j, hj⟩
  · exact Or.inl h2
  · exact Or.inr (Or.inl h3)
  · refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (h | h) | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

lemma ck_mem_Akn_succ (k : ℕ) : ck k ∈ Akn (k + 1) := by
  show ck k ∈ Akn k ∪ stage k
  exact Set.mem_union_right _ (Set.mem_union_left _ (Set.mem_union_left _ rfl))

lemma Bk_sub_Akn_succ (k : ℕ) : Bk k ⊆ Akn (k + 1) := by
  intro x hx; show x ∈ Akn k ∪ stage k
  exact Set.mem_union_right _ (Set.mem_union_left _ (Set.mem_union_right _ hx))

lemma Fk_sub_Akn_succ (k : ℕ) : Fk k ⊆ Akn (k + 1) := by
  intro x hx; show x ∈ Akn k ∪ stage k
  exact Set.mem_union_right _ (Set.mem_union_right _ hx)

lemma pair_cover {p q r s x : ℕ} {S : Set ℕ}
    (hSp : Icc p q ⊆ S) (hSr : Icc r s ⊆ S)
    (h1 : p + r ≤ x) (h2 : x ≤ q + s) (hpq : p ≤ q) (hrs : r ≤ s) :
    x ∈ S + S := by
  set b := max r (x - q) with hb
  have ha : x - b ∈ Icc p q := by rw [mem_Icc]; omega
  have hbmem : b ∈ Icc r s := by rw [mem_Icc]; omega
  exact Set.mem_add.mpr ⟨x - b, hSp ha, b, hSr hbmem, by omega⟩

lemma hI (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    show x ∈ Akn 0 ∪ stage 0
    refine Set.mem_union_left _ ?_
    simp only [Akn, Set.mem_insert_iff, Set.mem_singleton_iff]
    omega
  | succ j =>
    intro x hx
    rw [mem_Icc, Q_succ] at hx
    have hxF : x ∈ Fk j := by rw [Fk, mem_Icc]; omega
    exact Akn_mono (j + 1) (Fk_sub_Akn_succ j hxF)

lemma Hcov (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    obtain ⟨hx1, hx2⟩ := hx
    have h2 : (2 : ℕ) ∈ Akn 1 := by
      show (2 : ℕ) ∈ Akn 0 ∪ stage 0
      exact Set.mem_union_left _ (by simp [Akn])
    have h3 : (3 : ℕ) ∈ Akn 1 := by
      show (3 : ℕ) ∈ Akn 0 ∪ stage 0
      exact Set.mem_union_left _ (by simp [Akn])
    interval_cases x
    · exact Set.mem_add.mpr ⟨2, h2, 2, h2, rfl⟩
    · exact Set.mem_add.mpr ⟨2, h2, 3, h3, rfl⟩
    · exact Set.mem_add.mpr ⟨3, h3, 3, h3, rfl⟩
  | succ k ih =>
    intro x hx
    rw [mem_Icc] at hx
    have hpos : 0 < Q k := Q_pos k
    rw [Q_succ] at hx
    have hI' : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 2) :=
      fun y hy => Akn_mono _ (hI k hy)
    have hc' : Icc (4 * Q k) (4 * Q k) ⊆ Akn (k + 2) := by
      intro y hy
      rw [mem_Icc] at hy
      have hyc : y = ck k := by rw [ck]; omega
      rw [hyc]; exact Akn_mono _ (ck_mem_Akn_succ k)
    have hB' : Icc (5 * Q k) (6 * Q k - 1) ⊆ Akn (k + 2) :=
      fun y hy => Akn_mono _ (Bk_sub_Akn_succ k hy)
    have hF' : Icc (10 * Q k - 1) (15 * Q k) ⊆ Akn (k + 2) :=
      fun y hy => Akn_mono _ (Fk_sub_Akn_succ k hy)
    by_cases c1 : x ≤ 6 * Q k
    · have hmem := ih (mem_Icc.mpr ⟨hx.1, c1⟩)
      exact Set.add_subset_add (Akn_mono (k + 1)) (Akn_mono (k + 1)) hmem
    by_cases c2 : x ≤ 7 * Q k
    · exact pair_cover hI' hc' (by omega) (by omega) (by omega) (by omega)
    by_cases c3 : x ≤ 9 * Q k - 1
    · exact pair_cover hI' hB' (by omega) (by omega) (by omega) (by omega)
    by_cases c4 : x ≤ 10 * Q k - 1
    · exact pair_cover hc' hB' (by omega) (by omega) (by omega) (by omega)
    by_cases c5 : x ≤ 12 * Q k - 2
    · exact pair_cover hB' hB' (by omega) (by omega) (by omega) (by omega)
    by_cases c6 : x ≤ 18 * Q k
    · exact pair_cover hI' hF' (by omega) (by omega) (by omega) (by omega)
    by_cases c7 : x ≤ 21 * Q k - 1
    · exact pair_cover hB' hF' (by omega) (by omega) (by omega) (by omega)
    · exact pair_cover hF' hF' (by omega) (by omega) (by omega) (by omega)

lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have hle : n ≤ 6 * Q n := le_trans (le_Q n) (by omega)
  have hmem : n ∈ Akn (n + 1) + Akn (n + 1) := Hcov n (mem_Icc.mpr ⟨hn, hle⟩)
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  exact ⟨a, Akn_sub_setA _ ha, b, Akn_sub_setA _ hb, hab⟩

lemma two_le_mem {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases mem_setA hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have hpj : 0 < Q j := Q_pos j
    rcases hj with h | h | h
    · rw [ck] at h; omega
    · rw [Bk, mem_Icc] at h; omega
    · rw [Fk, mem_Icc] at h; omega

lemma window_lemma (k x : ℕ) (hx : x ∈ setA) (hlo : 3 * Q k < x) (hhi : x < 10 * Q k) :
    x = ck k ∨ x ∈ Bk k ∨ x = 10 * Q k - 1 := by
  have hpk : 0 < Q k := Q_pos k
  rcases mem_setA hx with h2 | h3 | ⟨j, haj⟩
  · exfalso; omega
  · exfalso; omega
  · have hpj : 0 < Q j := Q_pos j
    have hb1 : 4 * Q j ≤ x := by
      rcases haj with h | h | h
      · rw [ck] at h; omega
      · rw [Bk, mem_Icc] at h; omega
      · rw [Fk, mem_Icc] at h; omega
    have hb2 : x ≤ 15 * Q j := by
      rcases haj with h | h | h
      · rw [ck] at h; omega
      · rw [Bk, mem_Icc] at h; omega
      · rw [Fk, mem_Icc] at h; omega
    rcases lt_trichotomy j k with hlt | hje | hgt
    · exfalso; have := Q_lt hlt; omega
    · rw [hje] at haj
      rcases haj with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · rw [Fk, mem_Icc] at h
        right; right; omega
    · exfalso; have := Q_lt hgt; omega

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) (a : ℕ) (ha : a ∈ setA) (b : ℕ) (hb : b ∈ setA)
    (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  rw [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hpk : 0 < Q k := Q_pos k
  have halt : a < 10 * Q k := by omega
  have hblt : b < 10 * Q k := by omega
  by_cases hale : a ≤ 3 * Q k
  · -- a ≤ 3Qk : whole branch is vacuous
    have hbgt : 3 * Q k < b := by omega
    rcases window_lemma k b hb hbgt hblt with hbc | hbB | hbF
    · exfalso; rw [ck] at hbc; omega
    · exfalso; rw [Bk, mem_Icc] at hbB; omega
    · exfalso; have := two_le_mem ha; omega
  · have hagt : 3 * Q k < a := by omega
    rcases window_lemma k a ha hagt halt with hac | haB | haF
    · left
      refine ⟨hac, ?_⟩
      rw [Bk, mem_Icc]; rw [ck] at hac; omega
    · rw [Bk, mem_Icc] at haB
      have hbgt : 3 * Q k < b := by omega
      rcases window_lemma k b hb hbgt hblt with hbc | hbB | hbF
      · right
        refine ⟨hbc, ?_⟩
        rw [Bk, mem_Icc]; omega
      · exfalso; rw [Bk, mem_Icc] at hbB; omega
      · exfalso; omega
    · exfalso; have := two_le_mem hb; omega

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n, n ∈ Jk k → n ∉ T + T := by
  intro n hnJ hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  rcases rigidity k n hnJ a (hT ha) b (hT hb) hab with ⟨hac, _⟩ | ⟨hbc, _⟩
  · exact hck (hac ▸ ha)
  · exact hck (hbc ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, basis, ?_⟩
  intro A₁ A₂ h1 h2 hcov hdisj hsyn
  obtain ⟨hsyn1, hsyn2⟩ := hsyn
  obtain ⟨C₁, hC1⟩ := hsyn1
  obtain ⟨C₂, hC2⟩ := hsyn2
  set k := C₁ + C₂ + 1 with hk
  have hQk : C₁ + C₂ < Q k := lt_of_lt_of_le (by omega) (le_Q k)
  have hckA : ck k ∈ setA := ck_mem_setA k
  rcases hcov (ck k) hckA with hin1 | hin2
  · -- ck k ∈ A₁
    have hck2 : ck k ∉ A₂ := by
      intro hmem
      have hcap : ck k ∈ A₁ ∩ A₂ := ⟨hin1, hmem⟩
      rw [hdisj] at hcap
      exact hcap
    obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
    rw [mem_Icc] at hmI
    have hmJ : m ∈ Jk k := by
      rw [Jk, mem_Ico]
      refine ⟨hmI.1, ?_⟩
      have : C₂ < Q k := by omega
      omega
    exact gap_lem k A₂ h2 hck2 m hmJ hmS
  · -- ck k ∈ A₂
    have hck1 : ck k ∉ A₁ := by
      intro hmem
      have hcap : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hin2⟩
      rw [hdisj] at hcap
      exact hcap
    obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
    rw [mem_Icc] at hmI
    have hmJ : m ∈ Jk k := by
      rw [Jk, mem_Ico]
      refine ⟨hmI.1, ?_⟩
      have : C₁ < Q k := by omega
      omega
    exact gap_lem k A₁ h1 hck1 m hmJ hmS

end Erdos741OAI

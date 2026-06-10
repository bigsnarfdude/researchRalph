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
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := by simp only [Q]; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by simp only [Q, pow_succ]; ring

lemma Q_le {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  simp only [Q]; exact Nat.pow_le_pow_right (by norm_num) h

lemma five_mul_le {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have hjk : j + 1 ≤ k := by omega
  calc 5 * Q j = Q (j + 1) := (Q_succ j).symm
    _ ≤ Q k := Q_le hjk

lemma five_mul_ge {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have hkj : k + 1 ≤ j := by omega
  calc 5 * Q k = Q (k + 1) := (Q_succ k).symm
    _ ≤ Q j := Q_le hkj

lemma lt_Q (n : ℕ) : n < Q n := by
  induction n with
  | zero => exact Q_pos 0
  | succ m ih =>
    have hs : Q (m + 1) = 5 * Q m := Q_succ m
    rw [hs]
    have hp := Q_pos m
    omega

/-! ## Membership helpers -/

lemma ck_mem (k : ℕ) : ck k ∈ setA := by
  unfold setA
  exact Set.mem_union_right _
    (Set.mem_iUnion.mpr ⟨k, Set.mem_union_left _
      (Set.mem_union_left _ (Set.mem_singleton_iff.mpr rfl))⟩)

lemma Bk_sub (k : ℕ) : Bk k ⊆ setA := by
  intro x hx
  unfold setA
  exact Set.mem_union_right _
    (Set.mem_iUnion.mpr ⟨k, Set.mem_union_left _ (Set.mem_union_right _ hx)⟩)

lemma Fk_sub (k : ℕ) : Fk k ⊆ setA := by
  intro x hx
  unfold setA
  exact Set.mem_union_right _ (Set.mem_iUnion.mpr ⟨k, Set.mem_union_right _ hx⟩)

lemma I_sub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ setA := by
  intro x hx
  rw [Set.mem_Icc] at hx
  obtain ⟨hx1, hx2⟩ := hx
  cases k with
  | zero =>
    simp only [Q, pow_zero, mul_one] at hx1 hx2
    unfold setA
    apply Set.mem_union_left
    interval_cases x <;> simp
  | succ m =>
    apply Fk_sub m
    simp only [Fk, Set.mem_Icc]
    have hp := Q_pos m
    have hs : Q (m + 1) = 5 * Q m := Q_succ m
    rw [hs] at hx1 hx2
    constructor <;> omega

/-! ## Classification of elements of setA relative to a level k -/

lemma classification (k : ℕ) {x : ℕ} (hx : x ∈ setA) :
    (2 ≤ x ∧ x ≤ 3) ∨ (4 ≤ x ∧ x ≤ 3 * Q k) ∨ x = 4 * Q k ∨
    (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ x := by
  have hpk := Q_pos k
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_singleton_iff,
             ck, Bk, Fk, Set.mem_Icc, Set.mem_insert_iff] at hx
  rcases hx with (rfl | rfl) | ⟨j, hj⟩
  · omega
  · omega
  · have hpj := Q_pos j
    rcases hj with (hck | hb) | hf
    · rcases lt_trichotomy j k with hlt | heq | hgt
      · have h5 := five_mul_le hlt; omega
      · rw [heq] at hck; omega
      · have h5 := five_mul_ge hgt; omega
    · obtain ⟨hb1, hb2⟩ := hb
      rcases lt_trichotomy j k with hlt | heq | hgt
      · have h5 := five_mul_le hlt; omega
      · rw [heq] at hb1 hb2; omega
      · have h5 := five_mul_ge hgt; omega
    · obtain ⟨hf1, hf2⟩ := hf
      rcases lt_trichotomy j k with hlt | heq | hgt
      · have h5 := five_mul_le hlt; omega
      · rw [heq] at hf1 hf2; omega
      · have h5 := five_mul_ge hgt; omega

/-! ## Basis: setA + setA covers all n ≥ 4 -/

lemma basis_cover (k : ℕ) : Icc 4 (6 * Q k) ⊆ setA + setA := by
  induction k with
  | zero =>
    intro n hn
    rw [Set.mem_Icc] at hn
    obtain ⟨hn1, hn2⟩ := hn
    simp only [Q, pow_zero, mul_one] at hn2
    have h2 : (2 : ℕ) ∈ setA := by
      unfold setA; exact Set.mem_union_left _ (Set.mem_insert 2 {3})
    have h3 : (3 : ℕ) ∈ setA := by
      unfold setA; exact Set.mem_union_left _ (Set.mem_insert_of_mem 2 rfl)
    interval_cases n
    · exact Set.mem_add.mpr ⟨2, h2, 2, h2, rfl⟩
    · exact Set.mem_add.mpr ⟨2, h2, 3, h3, rfl⟩
    · exact Set.mem_add.mpr ⟨3, h3, 3, h3, rfl⟩
  | succ m ih =>
    intro n hn
    rw [Set.mem_Icc] at hn
    have hq := Q_pos m
    have hsucc : Q (m + 1) = 5 * Q m := Q_succ m
    rw [hsucc] at hn
    by_cases hsmall : n ≤ 6 * Q m
    · exact ih (Set.mem_Icc.mpr ⟨hn.1, hsmall⟩)
    · push_neg at hsmall
      rw [Set.mem_add]
      by_cases h1 : n ≤ 7 * Q m
      · exact ⟨4 * Q m, ck_mem m, n - 4 * Q m,
          I_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h2 : n ≤ 8 * Q m
      · exact ⟨5 * Q m, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 5 * Q m, I_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h3 : n ≤ 9 * Q m - 1
      · exact ⟨3 * Q m, I_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 3 * Q m, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h4 : n ≤ 10 * Q m - 1
      · exact ⟨4 * Q m, ck_mem m, n - 4 * Q m,
          Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h5 : n ≤ 11 * Q m - 1
      · exact ⟨5 * Q m, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 5 * Q m, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h6 : n ≤ 12 * Q m - 2
      · exact ⟨6 * Q m - 1, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - (6 * Q m - 1), Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h7 : n ≤ 17 * Q m
      · exact ⟨2 * Q m, I_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 2 * Q m, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h8 : n ≤ 18 * Q m
      · exact ⟨3 * Q m, I_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 3 * Q m, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h9 : n ≤ 20 * Q m
      · exact ⟨5 * Q m, Bk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 5 * Q m, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      by_cases h10 : n ≤ 25 * Q m - 1
      · exact ⟨10 * Q m - 1, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - (10 * Q m - 1), Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
      · exact ⟨15 * Q m, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩),
          n - 15 * Q m, Fk_sub m (Set.mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩

lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have hlt := lt_Q n
  have hq := Q_pos n
  have hmem : n ∈ Icc 4 (6 * Q n) := Set.mem_Icc.mpr ⟨hn, by omega⟩
  have h := basis_cover n hmem
  rwa [Set.mem_add] at h

/-! ## Rigidity and the gap argument -/

lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hpk := Q_pos k
  simp only [Jk, Set.mem_Ico] at hab
  obtain ⟨hlo, hhi⟩ := hab
  have hca := classification k ha
  have hcb := classification k hb
  rcases hca with ha1 | ha2 | ha3 | ha4 | ha5 <;>
    rcases hcb with hb1 | hb2 | hb3 | hb4 | hb5 <;>
      first
        | (left; exact ⟨ha3, Set.mem_Icc.mpr ⟨hb4.1, hb4.2⟩⟩)
        | (right; exact ⟨hb3, Set.mem_Icc.mpr ⟨ha4.1, ha4.2⟩⟩)
        | (exfalso; omega)

lemma gap (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [Set.eq_empty_iff_forall_notMem]
  intro x hx
  rw [Set.mem_inter_iff] at hx
  obtain ⟨hxJ, hxT⟩ := hx
  rw [Set.mem_add] at hxT
  obtain ⟨a, ha, b, hb, hab⟩ := hxT
  have ha' : a ∈ setA := hT ha
  have hb' : b ∈ setA := hT hb
  have hr := rigidity ha' hb' (by rw [hab]; exact hxJ)
  rcases hr with ⟨hac, _⟩ | ⟨hbc, _⟩
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
  · intro n hn; exact basis n hn
  · intro A₁ A₂ h1 h2 hcover hdisj
    rintro ⟨hs1, hs2⟩
    obtain ⟨C₁, hsyn1⟩ := hs1
    obtain ⟨C₂, hsyn2⟩ := hs2
    set k := C₁ + C₂ + 1 with hk
    have hQk : C₁ + C₂ < Q k := by have h := lt_Q k; omega
    have hpk := Q_pos k
    have hck_in : ck k ∈ setA := ck_mem k
    rcases hcover (ck k) hck_in with hc1 | hc2
    · have hnot2 : ck k ∉ A₂ := by
        intro hmem
        have hcon : ck k ∈ A₁ ∩ A₂ := ⟨hc1, hmem⟩
        rw [hdisj] at hcon; simp at hcon
      have hgap := gap k A₂ h2 hnot2
      obtain ⟨m, hmAdd, hmIcc⟩ := hsyn2 (9 * Q k)
      rw [Set.mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        simp only [Jk, Set.mem_Ico]; exact ⟨hmIcc.1, by omega⟩
      have hfin : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hmJ hmAdd
      rw [hgap] at hfin; simp at hfin
    · have hnot1 : ck k ∉ A₁ := by
        intro hmem
        have hcon : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hc2⟩
        rw [hdisj] at hcon; simp at hcon
      have hgap := gap k A₁ h1 hnot1
      obtain ⟨m, hmAdd, hmIcc⟩ := hsyn1 (9 * Q k)
      rw [Set.mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        simp only [Jk, Set.mem_Ico]; exact ⟨hmIcc.1, by omega⟩
      have hfin : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter hmJ hmAdd
      rw [hgap] at hfin; simp at hfin

end Erdos741OAI

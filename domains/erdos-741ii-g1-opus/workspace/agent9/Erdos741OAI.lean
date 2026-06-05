import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- ## Construction

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def Sk (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, Sk k

-- ## Basic facts about Q

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; positivity

lemma five_Q (k : ℕ) : 5 * Q k = Q (k + 1) := by
  unfold Q; rw [pow_succ]; ring

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => exact Nat.zero_le _
  | succ n ih =>
    have h := (five_Q n).symm
    have hp := Q_pos n
    omega

-- ## Membership helpers

lemma subck (k : ℕ) : Icc (4 * Q k) (4 * Q k) ⊆ setA := by
  intro x hx
  rw [mem_Icc] at hx
  have : x = 4 * Q k := by omega
  subst this
  exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

lemma subBk (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ setA := by
  intro x hx
  exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inr hx)⟩)

lemma subFk (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ setA := by
  intro x hx
  exact Or.inr (mem_iUnion.mpr ⟨k, Or.inr hx⟩)

lemma subI0 : Icc 2 3 ⊆ setA := by
  intro x hx
  rw [mem_Icc] at hx
  have hx2 : x = 2 ∨ x = 3 := by omega
  rcases hx2 with rfl | rfl
  · exact Or.inl (Or.inl rfl)
  · exact Or.inl (Or.inr rfl)

lemma ck_mem (k : ℕ) : ck k ∈ setA :=
  Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

-- ## The pair-cover helper

lemma pc (L1 H1 L2 H2 x : ℕ)
    (sub1 : Icc L1 H1 ⊆ setA) (sub2 : Icc L2 H2 ⊆ setA)
    (hL1 : L1 ≤ H1) (hL2 : L2 ≤ H2) (h1 : L1 + L2 ≤ x) (h2 : x ≤ H1 + H2) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = x := by
  by_cases hc : x ≤ H1 + L2
  · exact ⟨x - L2, sub1 (mem_Icc.mpr ⟨by omega, by omega⟩),
           L2, sub2 (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
  · exact ⟨H1, sub1 (mem_Icc.mpr ⟨by omega, by omega⟩),
           x - H1, sub2 (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩

-- ## The basis property

lemma basis : ∀ k x, 4 ≤ x → x ≤ 30 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = x := by
  intro k
  induction k with
  | zero =>
    intro x hx4 hxhi
    have h0 : Q 0 = 1 := rfl
    by_cases c1 : x ≤ 6
    · exact pc 2 3 2 3 x subI0 subI0 (by omega) (by omega) (by omega) (by omega)
    · by_cases c2 : x ≤ 7
      · exact pc 2 3 (4 * Q 0) (4 * Q 0) x subI0 (subck 0)
          (by omega) (by omega) (by omega) (by omega)
      · by_cases c3 : x ≤ 8
        · exact pc 2 3 (5 * Q 0) (6 * Q 0 - 1) x subI0 (subBk 0)
            (by omega) (by omega) (by omega) (by omega)
        · by_cases c4 : x ≤ 9
          · exact pc (4 * Q 0) (4 * Q 0) (5 * Q 0) (6 * Q 0 - 1) x (subck 0) (subBk 0)
              (by omega) (by omega) (by omega) (by omega)
          · by_cases c5 : x ≤ 10
            · exact pc (5 * Q 0) (6 * Q 0 - 1) (5 * Q 0) (6 * Q 0 - 1) x (subBk 0) (subBk 0)
                (by omega) (by omega) (by omega) (by omega)
            · by_cases c6 : x ≤ 18
              · exact pc 2 3 (10 * Q 0 - 1) (15 * Q 0) x subI0 (subFk 0)
                  (by omega) (by omega) (by omega) (by omega)
              · exact pc (10 * Q 0 - 1) (15 * Q 0) (10 * Q 0 - 1) (15 * Q 0) x (subFk 0) (subFk 0)
                  (by omega) (by omega) (by omega) (by omega)
  | succ k ih =>
    intro x hx4 hxhi
    by_cases hsmall : x ≤ 30 * Q k
    · exact ih x hx4 hsmall
    · have hqk := Q_pos k
      have hqk1 := Q_pos (k + 1)
      have hq' : Q (k + 1) = 5 * Q k := (five_Q k).symm
      by_cases c1 : x ≤ 7 * Q (k + 1)
      · exact pc (10 * Q k - 1) (15 * Q k) (4 * Q (k + 1)) (4 * Q (k + 1)) x
          (subFk k) (subck (k + 1)) (by omega) (by omega) (by omega) (by omega)
      · by_cases c2 : x ≤ 9 * Q (k + 1) - 1
        · exact pc (10 * Q k - 1) (15 * Q k) (5 * Q (k + 1)) (6 * Q (k + 1) - 1) x
            (subFk k) (subBk (k + 1)) (by omega) (by omega) (by omega) (by omega)
        · by_cases c3 : x ≤ 10 * Q (k + 1) - 1
          · exact pc (4 * Q (k + 1)) (4 * Q (k + 1)) (5 * Q (k + 1)) (6 * Q (k + 1) - 1) x
              (subck (k + 1)) (subBk (k + 1)) (by omega) (by omega) (by omega) (by omega)
          · by_cases c4 : x ≤ 12 * Q (k + 1) - 2
            · exact pc (5 * Q (k + 1)) (6 * Q (k + 1) - 1) (5 * Q (k + 1)) (6 * Q (k + 1) - 1) x
                (subBk (k + 1)) (subBk (k + 1)) (by omega) (by omega) (by omega) (by omega)
            · by_cases c5 : x ≤ 18 * Q (k + 1)
              · exact pc (10 * Q k - 1) (15 * Q k) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) x
                  (subFk k) (subFk (k + 1)) (by omega) (by omega) (by omega) (by omega)
              · by_cases c6 : x ≤ 21 * Q (k + 1) - 1
                · exact pc (5 * Q (k + 1)) (6 * Q (k + 1) - 1) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) x
                    (subBk (k + 1)) (subFk (k + 1)) (by omega) (by omega) (by omega) (by omega)
                · exact pc (10 * Q (k + 1) - 1) (15 * Q (k + 1)) (10 * Q (k + 1) - 1) (15 * Q (k + 1)) x
                    (subFk (k + 1)) (subFk (k + 1)) (by omega) (by omega) (by omega) (by omega)

-- ## Classification (locate) lemma

lemma locate (x : ℕ) (hx : x ∈ setA) (k : ℕ) :
    2 ≤ x ∧ (x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ x) := by
  have hqk := Q_pos k
  simp only [setA, Sk, ck, Bk, Fk, mem_union, mem_iUnion, mem_singleton_iff,
    mem_insert_iff, mem_Icc] at hx
  rcases hx with (rfl | rfl) | ⟨j, hj⟩
  · exact ⟨by norm_num, Or.inl (by omega)⟩
  · exact ⟨by norm_num, Or.inl (by omega)⟩
  · have hqj := Q_pos j
    rcases lt_trichotomy j k with hlt | hje | hgt
    · have h5 : 5 * Q j ≤ Q k := by rw [five_Q j]; exact Q_mono (by omega)
      exact ⟨by omega, by omega⟩
    · rw [hje] at hj
      exact ⟨by omega, by omega⟩
    · have h6 : 5 * Q k ≤ Q j := by rw [five_Q k]; exact Q_mono (by omega)
      exact ⟨by omega, by omega⟩

-- ## Rigidity

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) (a : ℕ) (ha : a ∈ setA)
    (b : ℕ) (hb : b ∈ setA) (hab : a + b = n) : a = ck k ∨ b = ck k := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨ha2, hca⟩ := locate a ha k
  obtain ⟨hb2, hcb⟩ := locate b hb k
  have hqk := Q_pos k
  simp only [ck]
  rcases hca with hca | hca | hca | hca
  · rcases hcb with hcb | hcb | hcb | hcb
    · omega
    · exact Or.inr hcb
    · omega
    · omega
  · exact Or.inl hca
  · rcases hcb with hcb | hcb | hcb | hcb
    · omega
    · exact Or.inr hcb
    · omega
    · omega
  · rcases hcb with hcb | hcb | hcb | hcb
    · omega
    · exact Or.inr hcb
    · omega
    · omega

-- ## Gap lemma

lemma gap (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [Set.eq_empty_iff_forall_notMem]
  intro n hn
  rw [mem_inter_iff] at hn
  obtain ⟨hnJ, hnTT⟩ := hn
  rw [Set.mem_add] at hnTT
  obtain ⟨a, haT, b, hbT, hab⟩ := hnTT
  rcases rigidity k n hnJ a (hT haT) b (hT hbT) hab with h | h
  · rw [h] at haT; exact hck haT
  · rw [h] at hbT; exact hck hbT

-- ## Main theorem

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
    exact basis n n hn (by have h := n_le_Q n; omega)
  · rintro A₁ A₂ hA1 hA2 hcover hdisj ⟨⟨C₁, hs1⟩, ⟨C₂, hs2⟩⟩
    set k := max C₁ C₂ + 1 with hk
    have hQk1 : C₁ < Q k := by
      have h := n_le_Q k; have := le_max_left C₁ C₂; omega
    have hQk2 : C₂ < Q k := by
      have h := n_le_Q k; have := le_max_right C₁ C₂; omega
    rcases hcover (ck k) (ck_mem k) with hc1 | hc2
    · have hnotA2 : ck k ∉ A₂ := by
        intro hmem
        have hcontra : ck k ∈ A₁ ∩ A₂ := ⟨hc1, hmem⟩
        rw [hdisj] at hcontra
        exact hcontra
      have hgap := gap k A₂ hA2 hnotA2
      obtain ⟨m, hmS, hmI⟩ := hs2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra
      exact hcontra
    · have hnotA1 : ck k ∉ A₁ := by
        intro hmem
        have hcontra : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hc2⟩
        rw [hdisj] at hcontra
        exact hcontra
      have hgap := gap k A₁ hA1 hnotA1
      obtain ⟨m, hmS, hmI⟩ := hs1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra
      exact hcontra

end Erdos741OAI

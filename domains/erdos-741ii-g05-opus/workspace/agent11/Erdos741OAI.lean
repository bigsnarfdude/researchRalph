import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k

def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k, ({4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k))

lemma Q_pos (k : ℕ) : 1 ≤ Q k := by unfold Q; exact Nat.one_le_pow _ _ (by norm_num)

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by unfold Q; rw [pow_succ]; ring

lemma n_lt_Q (n : ℕ) : n < Q n := by
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : (2 : ℕ) ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  unfold Q; omega

-- membership helpers
lemma two_mem : (2 : ℕ) ∈ setA := by apply Set.mem_union_left; exact Or.inl rfl
lemma three_mem : (3 : ℕ) ∈ setA := by apply Set.mem_union_left; exact Or.inr rfl
lemma p_mem (k : ℕ) : 4 * Q k ∈ setA := by
  apply Set.mem_union_right; rw [mem_iUnion]; exact ⟨k, Or.inl (Or.inl rfl)⟩
lemma M_mem (k : ℕ) {x : ℕ} (h : x ∈ Icc (5 * Q k) (6 * Q k - 1)) : x ∈ setA := by
  apply Set.mem_union_right; rw [mem_iUnion]; exact ⟨k, Or.inl (Or.inr h)⟩
lemma H_mem (k : ℕ) {x : ℕ} (h : x ∈ Icc (10 * Q k - 1) (15 * Q k)) : x ∈ setA := by
  apply Set.mem_union_right; rw [mem_iUnion]; exact ⟨k, Or.inr h⟩

lemma interval_sum_cover {a b c d n : ℕ} (hab : a ≤ b) (hcd : c ≤ d)
    (h1 : a + c ≤ n) (h2 : n ≤ b + d) : n ∈ Icc a b + Icc c d := by
  rw [Set.mem_add]
  refine ⟨min b (n - c), ?_, n - min b (n - c), ?_, ?_⟩
  · simp only [mem_Icc]; omega
  · simp only [mem_Icc]; omega
  · omega

lemma sum_mem_setA {a b c d : ℕ} (hsa : Icc a b ⊆ setA) (hsc : Icc c d ⊆ setA)
    (hab : a ≤ b) (hcd : c ≤ d) {n : ℕ} (h1 : a + c ≤ n) (h2 : n ≤ b + d) :
    n ∈ setA + setA := by
  have hcov := interval_sum_cover hab hcd h1 h2
  rw [Set.mem_add] at hcov ⊢
  obtain ⟨i, hi, j, hj, hij⟩ := hcov
  exact ⟨i, hsa hi, j, hsc hj, hij⟩

lemma small_sub : Icc 2 3 ⊆ setA := by
  intro x hx; rw [mem_Icc] at hx; obtain ⟨h1, h2⟩ := hx
  interval_cases x
  · exact two_mem
  · exact three_mem

lemma p_sub (k : ℕ) : Icc (4 * Q k) (4 * Q k) ⊆ setA := by
  intro x hx; rw [mem_Icc] at hx
  have hx' : x = 4 * Q k := le_antisymm hx.2 hx.1
  rw [hx']; exact p_mem k

lemma M_sub (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ setA := fun _ hx => M_mem k hx

lemma H_sub (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ setA := fun _ hx => H_mem k hx

lemma high_cover : ∀ k, Icc (6 * Q k) (30 * Q k) ⊆ setA + setA := by
  intro k
  rcases k with _ | j
  · -- k = 0 : cover Icc 6 30
    have hq0 : Q 0 = 1 := rfl
    intro x hx; rw [mem_Icc] at hx
    by_cases h1 : x ≤ 7
    · exact sum_mem_setA (p_sub 0) small_sub (by omega) (by omega) (by omega) (by omega)
    by_cases h2 : x ≤ 8
    · exact sum_mem_setA (M_sub 0) small_sub (by omega) (by omega) (by omega) (by omega)
    by_cases h3 : x ≤ 9
    · exact sum_mem_setA (p_sub 0) (M_sub 0) (by omega) (by omega) (by omega) (by omega)
    by_cases h4 : x ≤ 10
    · exact sum_mem_setA (M_sub 0) (M_sub 0) (by omega) (by omega) (by omega) (by omega)
    by_cases h5 : x ≤ 18
    · exact sum_mem_setA small_sub (H_sub 0) (by omega) (by omega) (by omega) (by omega)
    · exact sum_mem_setA (H_sub 0) (H_sub 0) (by omega) (by omega) (by omega) (by omega)
  · -- k = j+1
    have hqs : Q (j + 1) = 5 * Q j := Q_succ j
    have hqp : 1 ≤ Q j := Q_pos j
    intro x hx; rw [mem_Icc] at hx
    by_cases h1 : x ≤ 35 * Q j
    · exact sum_mem_setA (p_sub (j+1)) (H_sub j) (by omega) (by omega) (by omega) (by omega)
    by_cases h2 : x ≤ 45 * Q j - 1
    · exact sum_mem_setA (M_sub (j+1)) (H_sub j) (by omega) (by omega) (by omega) (by omega)
    by_cases h3 : x ≤ 50 * Q j - 1
    · exact sum_mem_setA (p_sub (j+1)) (M_sub (j+1)) (by omega) (by omega) (by omega) (by omega)
    by_cases h4 : x ≤ 60 * Q j - 2
    · exact sum_mem_setA (M_sub (j+1)) (M_sub (j+1)) (by omega) (by omega) (by omega) (by omega)
    by_cases h5 : x ≤ 90 * Q j
    · exact sum_mem_setA (H_sub j) (H_sub (j+1)) (by omega) (by omega) (by omega) (by omega)
    by_cases h6 : x ≤ 105 * Q j - 1
    · exact sum_mem_setA (M_sub (j+1)) (H_sub (j+1)) (by omega) (by omega) (by omega) (by omega)
    · exact sum_mem_setA (H_sub (j+1)) (H_sub (j+1)) (by omega) (by omega) (by omega) (by omega)

lemma basis_cover : ∀ k, Icc 4 (6 * Q k) ⊆ setA + setA := by
  intro k
  induction k with
  | zero =>
    have hq0 : Q 0 = 1 := rfl
    intro x hx; rw [mem_Icc] at hx
    exact sum_mem_setA small_sub small_sub (by omega) (by omega) (by omega) (by omega)
  | succ k ih =>
    have hqs : Q (k + 1) = 5 * Q k := Q_succ k
    intro x hx; rw [mem_Icc] at hx
    by_cases hle : x ≤ 6 * Q k
    · exact ih (by rw [mem_Icc]; exact ⟨hx.1, hle⟩)
    · have hxhi : x ∈ Icc (6 * Q k) (30 * Q k) := by
        rw [mem_Icc]; exact ⟨by omega, by omega⟩
      exact high_cover k hxhi

lemma Q_small {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  calc 5 * Q j = Q (j + 1) := (Q_succ j).symm
    _ ≤ Q k := by unfold Q; exact Nat.pow_le_pow_right (by norm_num) (by omega)

lemma Q_big {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  calc 5 * Q k = Q (k + 1) := (Q_succ k).symm
    _ ≤ Q j := by unfold Q; exact Nat.pow_le_pow_right (by norm_num) (by omega)

lemma classify {a : ℕ} (ha : a ∈ setA) :
    a = 2 ∨ a = 3 ∨ ∃ j, (a = 4 * Q j ∨ (5 * Q j ≤ a ∧ a ≤ 6 * Q j - 1) ∨
      (10 * Q j - 1 ≤ a ∧ a ≤ 15 * Q j)) := by
  rcases ha with h | h
  · simp only [mem_insert_iff, mem_singleton_iff] at h
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
  · rw [mem_iUnion] at h
    obtain ⟨j, hj⟩ := h
    simp only [mem_union, mem_singleton_iff, mem_Icc] at hj
    refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (hj | hj) | hj
    · exact Or.inl hj
    · exact Or.inr (Or.inl hj)
    · exact Or.inr (Or.inr hj)

lemma two_le_setA {a : ℕ} (ha : a ∈ setA) : 2 ≤ a := by
  rcases classify ha with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have := Q_pos j
    rcases hj with h | h | h <;> omega

lemma band_of_setA {k a : ℕ} (ha : a ∈ setA) (h : a < 10 * Q k) :
    a ≤ 6 * Q k - 1 ∨ a = 10 * Q k - 1 := by
  have hqk : 1 ≤ Q k := Q_pos k
  rcases classify ha with hh | hh | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hs := Q_small hjk
      have hqj := Q_pos j
      left
      rcases hj with h1 | h1 | h1 <;> omega
    · subst hjk
      rcases hj with h1 | h1 | h1
      · left; omega
      · left; omega
      · right; omega
    · have hb := Q_big hjk
      have hqj := Q_pos j
      exfalso
      rcases hj with h1 | h1 | h1 <;> omega

lemma band3 {k a : ℕ} (ha : a ∈ setA) (h : a ≤ 6 * Q k - 1) :
    a ≤ 3 * Q k ∨ a = 4 * Q k ∨ (5 * Q k ≤ a ∧ a ≤ 6 * Q k - 1) := by
  have hqk : 1 ≤ Q k := Q_pos k
  rcases classify ha with hh | hh | ⟨j, hj⟩
  · left; omega
  · left; omega
  · rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hs := Q_small hjk
      have hqj := Q_pos j
      left
      rcases hj with h1 | h1 | h1 <;> omega
    · subst hjk
      rcases hj with h1 | h1 | h1
      · right; left; omega
      · right; right; omega
      · have hqj := Q_pos k; exfalso; omega
    · have hb := Q_big hjk
      have hqj := Q_pos j
      exfalso
      rcases hj with h1 | h1 | h1 <;> omega

lemma rigid (k : ℕ) {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hlo : 9 * Q k ≤ a + b) (hhi : a + b < 10 * Q k) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hqk : 1 ≤ Q k := Q_pos k
  have ha2 := two_le_setA ha
  have hb2 := two_le_setA hb
  have hca := band_of_setA ha (by omega : a < 10 * Q k)
  have hcb := band_of_setA hb (by omega : b < 10 * Q k)
  rcases hca with hca | hca
  · rcases hcb with hcb | hcb
    · rcases band3 ha hca with h3a | h4a | hMa
      · rcases band3 hb hcb with h3b | h4b | hMb
        · omega
        · omega
        · omega
      · exact Or.inl h4a
      · rcases band3 hb hcb with h3b | h4b | hMb
        · omega
        · exact Or.inr h4b
        · omega
    · omega
  · omega

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
      have h := n_lt_Q n
      simp only [mem_Icc]; omega
    have hsum := basis_cover n hmem
    rw [Set.mem_add] at hsum
    obtain ⟨a, ha, b, hb, hab⟩ := hsum
    exact ⟨a, ha, b, hb, hab⟩
  · sorry

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 1600000
set_option maxRecDepth 4000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Scale sequence `Q k = 5^k`. -/
def Q (k : ℕ) : ℕ := 5 ^ k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ]; ring

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b :=
  Nat.pow_le_pow_right (by norm_num) h

lemma n_lt_Q (n : ℕ) : n < Q n := by
  have h2 : n < 2 ^ n := Nat.lt_two_pow_self
  have h25 : (2 : ℕ) ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  exact lt_of_lt_of_le h2 h25

/-- The witness set. -/
def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k : ℕ,
    ({4 * Q k} ∪ Set.Icc (5 * Q k) (6 * Q k - 1) ∪ Set.Icc (10 * Q k - 1) (15 * Q k))

/-- Minkowski-sum cover helper: `Icc a1 b1 + Icc a2 b2 ⊇ Icc (a1+a2) (b1+b2)`. -/
lemma interval_sum_cover {a1 b1 a2 b2 n : ℕ}
    (hn1 : a1 + a2 ≤ n) (hn2 : n ≤ b1 + b2) (hab1 : a1 ≤ b1) (hab2 : a2 ≤ b2) :
    ∃ i ∈ Set.Icc a1 b1, ∃ j ∈ Set.Icc a2 b2, i + j = n := by
  by_cases h : n ≤ b1 + a2
  · refine ⟨n - a2, ?_, a2, ?_, ?_⟩
    · constructor <;> omega
    · constructor <;> omega
    · omega
  · refine ⟨b1, ?_, n - b1, ?_, ?_⟩
    · constructor <;> omega
    · constructor <;> omega
    · omega

end Erdos741OAI

namespace Erdos741OAI

-- membership helpers into setA
lemma mem_two : (2 : ℕ) ∈ setA := by
  left; left; rfl

lemma mem_three : (3 : ℕ) ∈ setA := by
  left; right; rfl

lemma mem_spine (k : ℕ) : 4 * Q k ∈ setA := by
  right
  rw [Set.mem_iUnion]
  exact ⟨k, by left; left; rfl⟩

lemma mem_band1 (k : ℕ) {x : ℕ} (hx : x ∈ Set.Icc (5 * Q k) (6 * Q k - 1)) : x ∈ setA := by
  right
  rw [Set.mem_iUnion]
  exact ⟨k, by left; right; exact hx⟩

lemma mem_band2 (k : ℕ) {x : ℕ} (hx : x ∈ Set.Icc (10 * Q k - 1) (15 * Q k)) : x ∈ setA := by
  right
  rw [Set.mem_iUnion]
  exact ⟨k, by right; exact hx⟩

/-- Coverage: every `n ∈ [4, 6·Q(k+1)]` is a sum of two elements of `setA`. -/
lemma cover : ∀ k n, 4 ≤ n → n ≤ 6 * Q (k + 1) →
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro k
  induction k with
  | zero =>
    intro n h4 hub
    have hQ0 : Q 0 = 1 := by norm_num [Q]
    have hQ1 : Q 1 = 5 := by norm_num [Q]
    rw [hQ1] at hub
    have m2 : (2 : ℕ) ∈ setA := mem_two
    have m3 : (3 : ℕ) ∈ setA := mem_three
    have m4 : (4 : ℕ) ∈ setA := by simpa [hQ0] using mem_spine 0
    have m5 : (5 : ℕ) ∈ setA := by apply mem_band1 0; simp [hQ0]
    by_cases h10 : n ≤ 10
    · interval_cases n
      · exact ⟨2, m2, 2, m2, rfl⟩
      · exact ⟨2, m2, 3, m3, rfl⟩
      · exact ⟨3, m3, 3, m3, rfl⟩
      · exact ⟨3, m3, 4, m4, rfl⟩
      · exact ⟨4, m4, 4, m4, rfl⟩
      · exact ⟨4, m4, 5, m5, rfl⟩
      · exact ⟨5, m5, 5, m5, rfl⟩
    · push_neg at h10
      by_cases h17 : n ≤ 17
      · exact ⟨2, m2, n - 2, mem_band2 0 (by simp only [hQ0]; constructor <;> omega), by omega⟩
      · push_neg at h17
        obtain ⟨i, hi, j, hj, hij⟩ :=
          interval_sum_cover (a1 := 10 * Q 0 - 1) (b1 := 15 * Q 0)
            (a2 := 10 * Q 0 - 1) (b2 := 15 * Q 0) (n := n)
            (by simp only [hQ0]; omega) (by simp only [hQ0]; omega)
            (by simp only [hQ0]; omega) (by simp only [hQ0]; omega)
        exact ⟨i, mem_band2 0 hi, j, mem_band2 0 hj, hij⟩
  | succ k ih =>
    intro n h4 hub
    have hq1 : 1 ≤ Q k := Q_pos k
    have hsucc : Q (k + 1) = 5 * Q k := Q_succ k
    have hsucc2 : Q (k + 1 + 1) = 25 * Q k := by rw [Q_succ, Q_succ]; ring
    by_cases hsmall : n ≤ 6 * Q (k + 1)
    · exact ih n h4 hsmall
    · push_neg at hsmall
      set q := Q k with hq
      -- now 30q < n ≤ 150q
      by_cases c1 : n ≤ 35 * q
      · -- I1 : {20q} + band2 k = [30q-1, 35q]
        refine ⟨4 * Q (k + 1), mem_spine (k + 1), n - 4 * Q (k + 1),
          mem_band2 k ?_, by omega⟩
        constructor <;> omega
      · push_neg at c1
        by_cases c2 : n ≤ 45 * q - 1
        · -- I2 : band1 (k+1) + band2 k
          obtain ⟨i, hi, j, hj, hij⟩ :=
            interval_sum_cover (a1 := 5 * Q (k + 1)) (b1 := 6 * Q (k + 1) - 1)
              (a2 := 10 * Q k - 1) (b2 := 15 * Q k) (n := n)
              (by omega) (by omega) (by omega) (by omega)
          exact ⟨i, mem_band1 (k + 1) hi, j, mem_band2 k hj, hij⟩
        · push_neg at c2
          by_cases c3 : n ≤ 50 * q - 1
          · -- I3 : {20q} + band1 (k+1)
            refine ⟨4 * Q (k + 1), mem_spine (k + 1), n - 4 * Q (k + 1),
              mem_band1 (k + 1) ?_, by omega⟩
            constructor <;> omega
          · push_neg at c3
            by_cases c4 : n ≤ 60 * q - 2
            · -- I4 : band1 (k+1) + band1 (k+1)
              obtain ⟨i, hi, j, hj, hij⟩ :=
                interval_sum_cover (a1 := 5 * Q (k + 1)) (b1 := 6 * Q (k + 1) - 1)
                  (a2 := 5 * Q (k + 1)) (b2 := 6 * Q (k + 1) - 1) (n := n)
                  (by omega) (by omega) (by omega) (by omega)
              exact ⟨i, mem_band1 (k + 1) hi, j, mem_band1 (k + 1) hj, hij⟩
            · push_neg at c4
              by_cases c5 : n ≤ 81 * q - 1
              · -- I5 : band1 k + band2 (k+1)
                obtain ⟨i, hi, j, hj, hij⟩ :=
                  interval_sum_cover (a1 := 5 * Q k) (b1 := 6 * Q k - 1)
                    (a2 := 10 * Q (k + 1) - 1) (b2 := 15 * Q (k + 1)) (n := n)
                    (by omega) (by omega) (by omega) (by omega)
                exact ⟨i, mem_band1 k hi, j, mem_band2 (k + 1) hj, hij⟩
              · push_neg at c5
                by_cases c6 : n ≤ 105 * q - 1
                · -- I6 : band1 (k+1) + band2 (k+1)
                  obtain ⟨i, hi, j, hj, hij⟩ :=
                    interval_sum_cover (a1 := 5 * Q (k + 1)) (b1 := 6 * Q (k + 1) - 1)
                      (a2 := 10 * Q (k + 1) - 1) (b2 := 15 * Q (k + 1)) (n := n)
                      (by omega) (by omega) (by omega) (by omega)
                  exact ⟨i, mem_band1 (k + 1) hi, j, mem_band2 (k + 1) hj, hij⟩
                · push_neg at c6
                  -- I7 : band2 (k+1) + band2 (k+1) = [100q-2, 150q]
                  obtain ⟨i, hi, j, hj, hij⟩ :=
                    interval_sum_cover (a1 := 10 * Q (k + 1) - 1) (b1 := 15 * Q (k + 1))
                      (a2 := 10 * Q (k + 1) - 1) (b2 := 15 * Q (k + 1)) (n := n)
                      (by omega) (by omega) (by omega) (by omega)
                  exact ⟨i, mem_band2 (k + 1) hi, j, mem_band2 (k + 1) hj, hij⟩

lemma Q5lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have hh : Q (j + 1) ≤ Q k := Q_mono h
  rw [Q_succ] at hh; exact hh

lemma Q5gt {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have hh : Q (k + 1) ≤ Q j := Q_mono h
  rw [Q_succ] at hh; exact hh

/-- Every element of `setA` is `2`, `3`, or in one of the three stage-`j` pieces. -/
lemma classify (x : ℕ) (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨
      (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  simp only [setA, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff,
    Set.mem_iUnion, Set.mem_Icc] at hx
  rcases hx with (h2 | h3) | ⟨j, hj⟩
  · exact Or.inl h2
  · exact Or.inr (Or.inl h3)
  · refine Or.inr (Or.inr ⟨j, ?_⟩)
    rcases hj with (h | h) | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

/-- An element of `setA` below `10·Q k` is small (`≤ 3 Q k`) or one of stage `k`'s low pieces. -/
lemma elt_bound (x : ℕ) (hx : x ∈ setA) (k : ℕ) (hxk : x < 10 * Q k) :
    (2 ≤ x ∧ x ≤ 3 * Q k) ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨
      x = 10 * Q k - 1 := by
  have hQk : 1 ≤ Q k := Q_pos k
  rcases classify x hx with rfl | rfl | ⟨j, hj⟩
  · left; omega
  · left; omega
  · have hQj : 1 ≤ Q j := Q_pos j
    rcases hj with h4 | hb1 | hb2
    · rcases lt_trichotomy j k with hjk | rfl | hjk
      · have h5 := Q5lt hjk; left; omega
      · exact Or.inr (Or.inl h4)
      · have h5 := Q5gt hjk; exfalso; omega
    · rcases lt_trichotomy j k with hjk | rfl | hjk
      · have h5 := Q5lt hjk; left; omega
      · exact Or.inr (Or.inr (Or.inl hb1))
      · have h5 := Q5gt hjk; exfalso; omega
    · rcases lt_trichotomy j k with hjk | rfl | hjk
      · have h5 := Q5lt hjk; left; omega
      · right; right; right; omega
      · have h5 := Q5gt hjk; exfalso; omega

/-- Rigidity: any representation of an element of `[9 Q k, 10 Q k - 1]` as a sum of two
`setA` elements must use the spine point `4 Q k`. -/
lemma spine_in (k a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (h1 : 9 * Q k ≤ a + b) (h2 : a + b ≤ 10 * Q k - 1) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hQk : 1 ≤ Q k := Q_pos k
  have hba := elt_bound a ha k (by omega)
  have hbb := elt_bound b hb k (by omega)
  rcases hba with ha' | ha' | ha' | ha' <;> rcases hbb with hb' | hb' | hb' | hb' <;>
    first
      | (left; omega)
      | (right; omega)
      | omega

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · -- additive basis of order 2 for n ≥ 4
    intro n hn
    refine cover n n hn ?_
    have h1 := n_lt_Q n
    have h2 := Q_mono (show n ≤ n + 1 by omega)
    have h3 := Q_pos (n + 1)
    omega
  · -- rigidity
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    set k := C₁ + C₂ + 1 with hkdef
    have hkQ : k < Q k := n_lt_Q k
    have hspine : 4 * Q k ∈ setA := mem_spine k
    rcases hpart _ hspine with hsp | hsp
    · -- 4 Q k ∈ A₁, so A₂ + A₂ misses the window [9 Q k, 10 Q k - 1]
      obtain ⟨m, hmem, hicc⟩ := hC₂ (9 * Q k)
      rw [Set.mem_Icc] at hicc
      rw [Set.mem_add] at hmem
      obtain ⟨a, haA, b, hbA, hab⟩ := hmem
      have ha : a ∈ setA := hA₂ haA
      have hb : b ∈ setA := hA₂ hbA
      have hsp_in := spine_in k a b ha hb (by omega) (by omega)
      have h4 : 4 * Q k ∈ A₂ := by
        rcases hsp_in with h | h
        · rw [← h]; exact haA
        · rw [← h]; exact hbA
      have hcontra : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hsp, h4⟩
      rw [hdisj] at hcontra
      exact absurd hcontra (by simp)
    · -- 4 Q k ∈ A₂, symmetric
      obtain ⟨m, hmem, hicc⟩ := hC₁ (9 * Q k)
      rw [Set.mem_Icc] at hicc
      rw [Set.mem_add] at hmem
      obtain ⟨a, haA, b, hbA, hab⟩ := hmem
      have ha : a ∈ setA := hA₁ haA
      have hb : b ∈ setA := hA₁ hbA
      have hsp_in := spine_in k a b ha hb (by omega) (by omega)
      have h4 : 4 * Q k ∈ A₁ := by
        rcases hsp_in with h | h
        · rw [← h]; exact haA
        · rw [← h]; exact hbA
      have hcontra : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h4, hsp⟩
      rw [hdisj] at hcontra
      exact absurd hcontra (by simp)

end Erdos741OAI

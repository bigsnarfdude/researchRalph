import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k

def stage (k : ℕ) : Set ℕ :=
  {4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

-- Basic facts about Q -------------------------------------------------------

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; positivity

lemma Q_mono {a b : ℕ} (h : a ≤ b) : Q a ≤ Q b := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma Q_step {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have h1 : Q (j + 1) ≤ Q k := Q_mono (by omega)
  rw [Q_succ] at h1; exact h1

lemma lt_Q (n : ℕ) : n < Q n := by
  unfold Q
  calc n < 2 ^ n := Nat.lt_two_pow_self
    _ ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n

-- Membership injectors ------------------------------------------------------

lemma mem_setA_of_stage {x j : ℕ} (h : x ∈ stage j) : x ∈ setA := by
  simp only [setA, Set.mem_union, Set.mem_iUnion]
  exact Or.inr ⟨j, h⟩

lemma point_mem (k : ℕ) : 4 * Q k ∈ setA := by
  apply mem_setA_of_stage (j := k)
  unfold stage
  exact Set.mem_union_left _ (Set.mem_union_left _ rfl)

lemma mid_mem {k x : ℕ} (h : x ∈ Icc (5 * Q k) (6 * Q k - 1)) : x ∈ setA := by
  apply mem_setA_of_stage (j := k)
  simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc]
  rw [mem_Icc] at h
  exact Or.inl (Or.inr h)

lemma high_mem {k x : ℕ} (h : x ∈ Icc (10 * Q k - 1) (15 * Q k)) : x ∈ setA := by
  apply mem_setA_of_stage (j := k)
  simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc]
  rw [mem_Icc] at h
  exact Or.inr h

lemma mem2 : (2 : ℕ) ∈ setA := by
  unfold setA
  exact Set.mem_union_left _ (Set.mem_insert 2 {3})

lemma mem3 : (3 : ℕ) ∈ setA := by
  unfold setA
  exact Set.mem_union_left _ (Set.mem_insert_of_mem 2 rfl)

lemma mem4 : (4 : ℕ) ∈ setA := by
  have h := point_mem 0
  simpa [Q] using h

lemma mem5 : (5 : ℕ) ∈ setA := by
  apply mid_mem (k := 0)
  simp only [Q, pow_zero, mem_Icc]
  omega

lemma mem_high0 (x : ℕ) (hx9 : 9 ≤ x) (hx15 : x ≤ 15) : x ∈ setA := by
  apply high_mem (k := 0)
  simp only [Q, pow_zero, mem_Icc]
  omega

lemma small_mem {x : ℕ} (h : x ∈ Icc (2 : ℕ) 3) : x ∈ setA := by
  rw [mem_Icc] at h
  have hx : x = 2 ∨ x = 3 := by omega
  rcases hx with hx | hx
  · rw [hx]; exact mem2
  · rw [hx]; exact mem3

-- Classification ------------------------------------------------------------

lemma setA_loc {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, x ∈ stage j := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_insert_iff,
    Set.mem_singleton_iff] at hx
  rcases hx with (h | h) | ⟨j, hj⟩
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr ⟨j, hj⟩)

lemma stage_bounds {x j : ℕ} (h : x ∈ stage j) : 4 * Q j ≤ x ∧ x ≤ 15 * Q j := by
  have hp := Q_pos j
  simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at h
  rcases h with (h | h) | h
  · subst h; omega
  · omega
  · omega

lemma setA_ge_two {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  rcases setA_loc hx with h | h | ⟨j, hj⟩
  · omega
  · omega
  · have := stage_bounds hj; have := Q_pos j; omega

lemma stage_eq {x j k : ℕ} (hxs : x ∈ stage j) (h1 : 3 * Q k < x) (h2 : x < 20 * Q k) :
    j = k := by
  obtain ⟨hb1, hb2⟩ := stage_bounds hxs
  rcases lt_trichotomy j k with h | h | h
  · exfalso; have := Q_step h; omega
  · exact h
  · exfalso; have := Q_step h; omega

-- Rigidity ------------------------------------------------------------------

lemma rigid_le {a b k : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b ∈ Icc (9 * Q k) (10 * Q k - 1)) (hle : a ≤ b) : a = 4 * Q k := by
  have hp := Q_pos k
  rw [mem_Icc] at hab
  obtain ⟨hm1, hm2⟩ := hab
  have ha2 : 2 ≤ a := setA_ge_two ha
  have hb2 : 2 ≤ b := setA_ge_two hb
  have hbmid : 5 * Q k ≤ b ∧ b ≤ 6 * Q k - 1 := by
    have h2b : 9 * Q k ≤ 2 * b := by omega
    rcases setA_loc hb with hb_2 | hb_3 | ⟨j, hbj⟩
    · omega
    · omega
    · have hjk : j = k := stage_eq hbj (by omega) (by omega)
      subst hjk
      simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at hbj
      rcases hbj with (h | h) | h
      · omega
      · exact h
      · omega
  obtain ⟨hbm1, hbm2⟩ := hbmid
  rcases setA_loc ha with ha_2 | ha_3 | ⟨j, haj⟩
  · omega
  · omega
  · have hjk : j = k := stage_eq haj (by omega) (by omega)
    subst hjk
    simp only [stage, Set.mem_union, Set.mem_singleton_iff, Set.mem_Icc] at haj
    rcases haj with (h | h) | h
    · exact h
    · omega
    · omega

lemma rigid {a b k : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b ∈ Icc (9 * Q k) (10 * Q k - 1)) : a = 4 * Q k ∨ b = 4 * Q k := by
  by_cases h : a ≤ b
  · exact Or.inl (rigid_le ha hb hab h)
  · have h' : b ≤ a := by omega
    have hab' : b + a ∈ Icc (9 * Q k) (10 * Q k - 1) := by rw [Nat.add_comm]; exact hab
    exact Or.inr (rigid_le hb ha hab' h')

-- Minkowski cover of intervals ---------------------------------------------

lemma pair_mem {a b c d n : ℕ} (hab : a ≤ b) (hcd : c ≤ d) (h1 : a + c ≤ n)
    (h2 : n ≤ b + d) : ∃ x ∈ Icc a b, ∃ y ∈ Icc c d, x + y = n := by
  refine ⟨max a (n - d), ?_, n - max a (n - d), ?_, ?_⟩
  · rw [mem_Icc]; omega
  · rw [mem_Icc]; omega
  · omega

-- Basis -------------------------------------------------------------------

lemma cover_base : ∀ n, 4 ≤ n → n ≤ 30 → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n h1 h2
  interval_cases n
  · exact ⟨2, mem2, 2, mem2, by norm_num⟩
  · exact ⟨2, mem2, 3, mem3, by norm_num⟩
  · exact ⟨3, mem3, 3, mem3, by norm_num⟩
  · exact ⟨3, mem3, 4, mem4, by norm_num⟩
  · exact ⟨4, mem4, 4, mem4, by norm_num⟩
  · exact ⟨4, mem4, 5, mem5, by norm_num⟩
  · exact ⟨5, mem5, 5, mem5, by norm_num⟩
  · exact ⟨2, mem2, 9, mem_high0 9 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 10, mem_high0 10 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 11, mem_high0 11 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 12, mem_high0 12 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 13, mem_high0 13 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 14, mem_high0 14 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨2, mem2, 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 9, mem_high0 9 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 10, mem_high0 10 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 11, mem_high0 11 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 12, mem_high0 12 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 13, mem_high0 13 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 14, mem_high0 14 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨9, mem_high0 9 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨10, mem_high0 10 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨11, mem_high0 11 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨12, mem_high0 12 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨13, mem_high0 13 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨14, mem_high0 14 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩
  · exact ⟨15, mem_high0 15 (by norm_num) (by norm_num), 15, mem_high0 15 (by norm_num) (by norm_num), by norm_num⟩

lemma cover_band {k : ℕ} (hk : 1 ≤ k) {n : ℕ} (h1 : 6 * Q k < n) (h2 : n ≤ 30 * Q k) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
  have hQ : Q (m + 1) = 5 * Q m := Q_succ m
  have hp := Q_pos m
  by_cases c1 : n ≤ 35 * Q m
  · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 10 * Q m - 1) (b := 15 * Q m)
      (c := 4 * Q (m + 1)) (d := 4 * Q (m + 1)) (n := n) (by omega) (by omega) (by omega) (by omega)
    refine ⟨x, high_mem hx, y, ?_, hxy⟩
    rw [mem_Icc] at hy
    have : y = 4 * Q (m + 1) := by omega
    rw [this]; exact point_mem (m + 1)
  · by_cases c2 : n ≤ 45 * Q m - 1
    · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 10 * Q m - 1) (b := 15 * Q m)
        (c := 5 * Q (m + 1)) (d := 6 * Q (m + 1) - 1) (n := n) (by omega) (by omega) (by omega) (by omega)
      exact ⟨x, high_mem hx, y, mid_mem hy, hxy⟩
    · by_cases c3 : n ≤ 50 * Q m - 1
      · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 4 * Q (m + 1)) (b := 4 * Q (m + 1))
          (c := 5 * Q (m + 1)) (d := 6 * Q (m + 1) - 1) (n := n) (by omega) (by omega) (by omega) (by omega)
        refine ⟨x, ?_, y, mid_mem hy, hxy⟩
        rw [mem_Icc] at hx
        have : x = 4 * Q (m + 1) := by omega
        rw [this]; exact point_mem (m + 1)
      · by_cases c4 : n ≤ 60 * Q m - 2
        · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 5 * Q (m + 1)) (b := 6 * Q (m + 1) - 1)
            (c := 5 * Q (m + 1)) (d := 6 * Q (m + 1) - 1) (n := n) (by omega) (by omega) (by omega) (by omega)
          exact ⟨x, mid_mem hx, y, mid_mem hy, hxy⟩
        · by_cases c5 : n ≤ 75 * Q m + 3
          · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 2) (b := 3)
              (c := 10 * Q (m + 1) - 1) (d := 15 * Q (m + 1)) (n := n) (by omega) (by omega) (by omega) (by omega)
            exact ⟨x, small_mem hx, y, high_mem hy, hxy⟩
          · by_cases c6 : n ≤ 95 * Q m
            · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 4 * Q (m + 1)) (b := 4 * Q (m + 1))
                (c := 10 * Q (m + 1) - 1) (d := 15 * Q (m + 1)) (n := n) (by omega) (by omega) (by omega) (by omega)
              refine ⟨x, ?_, y, high_mem hy, hxy⟩
              rw [mem_Icc] at hx
              have : x = 4 * Q (m + 1) := by omega
              rw [this]; exact point_mem (m + 1)
            · by_cases c7 : n ≤ 105 * Q m - 1
              · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 5 * Q (m + 1)) (b := 6 * Q (m + 1) - 1)
                  (c := 10 * Q (m + 1) - 1) (d := 15 * Q (m + 1)) (n := n) (by omega) (by omega) (by omega) (by omega)
                exact ⟨x, mid_mem hx, y, high_mem hy, hxy⟩
              · obtain ⟨x, hx, y, hy, hxy⟩ := pair_mem (a := 10 * Q (m + 1) - 1) (b := 15 * Q (m + 1))
                  (c := 10 * Q (m + 1) - 1) (d := 15 * Q (m + 1)) (n := n) (by omega) (by omega) (by omega) (by omega)
                exact ⟨x, high_mem hx, y, high_mem hy, hxy⟩

lemma cover : ∀ k n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro k
  induction k with
  | zero =>
    intro n hn1 hn2
    have hq : Q 0 = 1 := by decide
    rw [hq] at hn2
    exact cover_base n hn1 (by omega)
  | succ m ih =>
    intro n hn1 hn2
    rcases Nat.eq_zero_or_pos m with hm0 | hmpos
    · subst hm0
      have hq : Q (0 + 1) = 5 := by decide
      rw [hq] at hn2
      exact cover_base n hn1 (by omega)
    · by_cases hsmall : n ≤ 6 * Q m
      · exact ih n hn1 hsmall
      · have hb2 : n ≤ 30 * Q m := by have hq := Q_succ m; omega
        exact cover_band (k := m) (by omega) (by omega) hb2

-- Main theorem --------------------------------------------------------------

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
    exact cover n n hn (by have := lt_Q n; omega)
  · intro A₁ A₂ hsub1 hsub2 hcov hdisj
    rintro ⟨hS1, hS2⟩
    unfold IsSyndetic at hS1 hS2
    obtain ⟨C₁, hC₁⟩ := hS1
    obtain ⟨C₂, hC₂⟩ := hS2
    set k := C₁ + C₂ + 1 with hk
    have hQk1 : C₁ < Q k := by have := lt_Q k; omega
    have hQk2 : C₂ < Q k := by have := lt_Q k; omega
    have hp := Q_pos k
    have h4 : 4 * Q k ∈ setA := point_mem k
    rcases hcov _ h4 with h1 | h2
    · obtain ⟨w, hwS, hwI⟩ := hC₂ (9 * Q k)
      rw [mem_Icc] at hwI
      rw [Set.mem_add] at hwS
      obtain ⟨a, ha, b, hb, hab⟩ := hwS
      have haA : a ∈ setA := hsub2 ha
      have hbA : b ∈ setA := hsub2 hb
      have hwJ : a + b ∈ Icc (9 * Q k) (10 * Q k - 1) := by
        rw [mem_Icc, hab]
        exact ⟨hwI.1, by have := hwI.2; omega⟩
      rcases rigid haA hbA hwJ with hx | hx
      · have hin : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h1, hx ▸ ha⟩
        rw [hdisj] at hin; exact hin
      · have hin : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h1, hx ▸ hb⟩
        rw [hdisj] at hin; exact hin
    · obtain ⟨w, hwS, hwI⟩ := hC₁ (9 * Q k)
      rw [mem_Icc] at hwI
      rw [Set.mem_add] at hwS
      obtain ⟨a, ha, b, hb, hab⟩ := hwS
      have haA : a ∈ setA := hsub1 ha
      have hbA : b ∈ setA := hsub1 hb
      have hwJ : a + b ∈ Icc (9 * Q k) (10 * Q k - 1) := by
        rw [mem_Icc, hab]
        exact ⟨hwI.1, by have := hwI.2; omega⟩
      rcases rigid haA hbA hwJ with hx | hx
      · have hin : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hx ▸ ha, h2⟩
        rw [hdisj] at hin; exact hin
      · have hin : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hx ▸ hb, h2⟩
        rw [hdisj] at hin; exact hin

end Erdos741OAI

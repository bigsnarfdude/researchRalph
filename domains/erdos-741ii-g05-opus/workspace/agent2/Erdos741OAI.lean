import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-- Geometric scale. -/
def Q (k : ℕ) : ℕ := 5 ^ k

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; positivity

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q; rw [pow_succ]; ring

lemma Q_le {i j : ℕ} (h : i ≤ j) : Q i ≤ Q j := by
  unfold Q; exact Nat.pow_le_pow_right (by norm_num) h

lemma Q_step_lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have h1 : j + 1 ≤ k := h
  have h2 : Q (j + 1) ≤ Q k := Q_le h1
  rw [Q_succ] at h2; exact h2

lemma Q_step_gt {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have h1 : k + 1 ≤ j := h
  have h2 : Q (k + 1) ≤ Q j := Q_le h1
  rw [Q_succ] at h2; exact h2

lemma Q_gt (n : ℕ) : n < Q n := by
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : (2 : ℕ) ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  unfold Q; omega

/-- The construction. -/
def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k, ({4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k))

lemma mem_setA_stage {z k : ℕ}
    (h : z ∈ ({4 * Q k} ∪ Icc (5 * Q k) (6 * Q k - 1) ∪ Icc (10 * Q k - 1) (15 * Q k))) :
    z ∈ setA :=
  Or.inr (Set.mem_iUnion.mpr ⟨k, h⟩)

lemma haI (k : ℕ) : Icc (4 * Q k) (4 * Q k) ⊆ setA := by
  intro z hz; rw [Set.mem_Icc] at hz
  have hzeq : z = 4 * Q k := le_antisymm hz.2 hz.1
  exact mem_setA_stage (Or.inl (Or.inl (Set.mem_singleton_iff.mpr hzeq)))

lemma hBsub (k : ℕ) : Icc (5 * Q k) (6 * Q k - 1) ⊆ setA := by
  intro z hz
  exact mem_setA_stage (Or.inl (Or.inr hz))

lemma hCsub (k : ℕ) : Icc (10 * Q k - 1) (15 * Q k) ⊆ setA := by
  intro z hz
  exact mem_setA_stage (Or.inr hz)

lemma hIsub (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ setA := by
  intro z hz; rw [Set.mem_Icc] at hz
  cases k with
  | zero =>
    have hq : Q 0 = 1 := by norm_num [Q]
    have h1 := hz.1; have h2 := hz.2
    rw [hq] at h1 h2
    have hzr : z = 2 ∨ z = 3 := by omega
    rcases hzr with h | h
    · exact Or.inl (Set.mem_insert_iff.mpr (Or.inl h))
    · exact Or.inl (Set.mem_insert_iff.mpr (Or.inr (Set.mem_singleton_iff.mpr h)))
  | succ j =>
    have hQ : Q (j + 1) = 5 * Q j := Q_succ j
    apply hCsub j
    rw [Set.mem_Icc]; omega

lemma elt_ge_two (z : ℕ) (hz : z ∈ setA) : 2 ≤ z := by
  simp only [setA, Set.mem_union, Set.mem_iUnion] at hz
  rcases hz with h23 | ⟨j, (hsing | hBmem) | hCmem⟩
  · rw [Set.mem_insert_iff, Set.mem_singleton_iff] at h23
    rcases h23 with h | h <;> omega
  · have hpj := Q_pos j
    rw [Set.mem_singleton_iff] at hsing; omega
  · have hpj := Q_pos j
    rw [Set.mem_Icc] at hBmem; omega
  · have hpj := Q_pos j
    rw [Set.mem_Icc] at hCmem; omega

lemma elt_le_or (k z : ℕ) (hz : z ∈ setA) (hlt : z < 10 * Q k) :
    z ≤ 3 * Q k ∨ z = 4 * Q k ∨ (5 * Q k ≤ z ∧ z ≤ 6 * Q k - 1) ∨ z = 10 * Q k - 1 := by
  have hpk := Q_pos k
  simp only [setA, Set.mem_union, Set.mem_iUnion] at hz
  rcases hz with h23 | ⟨j, (hsing | hBmem) | hCmem⟩
  · rw [Set.mem_insert_iff, Set.mem_singleton_iff] at h23
    rcases h23 with h | h <;> omega
  · have hpj := Q_pos j
    rw [Set.mem_singleton_iff] at hsing
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hs := Q_step_lt hjk; omega
    · have hQeq : Q j = Q k := by rw [hjk]
      omega
    · have hs := Q_step_gt hjk; omega
  · have hpj := Q_pos j
    rw [Set.mem_Icc] at hBmem
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hs := Q_step_lt hjk; omega
    · have hQeq : Q j = Q k := by rw [hjk]
      omega
    · have hs := Q_step_gt hjk; omega
  · have hpj := Q_pos j
    rw [Set.mem_Icc] at hCmem
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · have hs := Q_step_lt hjk; omega
    · have hQeq : Q j = Q k := by rw [hjk]
      omega
    · have hs := Q_step_gt hjk; omega

lemma sum_in_J_forces (k x y : ℕ) (hx : x ∈ setA) (hy : y ∈ setA)
    (hlo : 9 * Q k ≤ x + y) (hhi : x + y < 10 * Q k) : x = 4 * Q k ∨ y = 4 * Q k := by
  have hpk := Q_pos k
  have hx2 := elt_ge_two x hx
  have hy2 := elt_ge_two y hy
  have hxlt : x < 10 * Q k := by omega
  have hylt : y < 10 * Q k := by omega
  have bx := elt_le_or k x hx hxlt
  have byy := elt_le_or k y hy hylt
  rcases bx with hxa | hxb | hxc | hxd
  · rcases byy with hya | hyb | hyc | hyd
    · omega
    · exact Or.inr hyb
    · omega
    · omega
  · exact Or.inl hxb
  · rcases byy with hya | hyb | hyc | hyd
    · omega
    · exact Or.inr hyb
    · omega
    · omega
  · omega

lemma interval_sum_cover (a b c d n : ℕ) (hab : a ≤ b) (hcd : c ≤ d)
    (h1 : a + c ≤ n) (h2 : n ≤ b + d) :
    ∃ i j, (a ≤ i ∧ i ≤ b) ∧ (c ≤ j ∧ j ≤ d) ∧ i + j = n := by
  refine ⟨min b (n - c), n - min b (n - c), ⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_⟩ <;> omega

lemma pair_mem (a b c d n : ℕ) (hab : a ≤ b) (hcd : c ≤ d)
    (hI1 : Icc a b ⊆ setA) (hI2 : Icc c d ⊆ setA)
    (h1 : a + c ≤ n) (h2 : n ≤ b + d) : n ∈ setA + setA := by
  obtain ⟨i, j, ⟨hia, hib⟩, ⟨hjc, hjd⟩, hij⟩ := interval_sum_cover a b c d n hab hcd h1 h2
  rw [Set.mem_add]
  exact ⟨i, hI1 (Set.mem_Icc.mpr ⟨hia, hib⟩), j, hI2 (Set.mem_Icc.mpr ⟨hjc, hjd⟩), hij⟩

lemma basis_up (k : ℕ) : Icc 4 (6 * Q k) ⊆ setA + setA := by
  induction k with
  | zero =>
    intro n hn; rw [Set.mem_Icc] at hn
    have hq : Q 0 = 1 := by norm_num [Q]
    rw [hq] at hn
    have h2 : (2 : ℕ) ∈ setA := Or.inl (Set.mem_insert_iff.mpr (Or.inl rfl))
    have h3 : (3 : ℕ) ∈ setA :=
      Or.inl (Set.mem_insert_iff.mpr (Or.inr (Set.mem_singleton_iff.mpr rfl)))
    rw [Set.mem_add]
    rcases (by omega : n = 4 ∨ n = 5 ∨ n = 6) with h | h | h
    · exact ⟨2, h2, 2, h2, by omega⟩
    · exact ⟨2, h2, 3, h3, by omega⟩
    · exact ⟨3, h3, 3, h3, by omega⟩
  | succ k ih =>
    intro n hn; rw [Set.mem_Icc] at hn
    obtain ⟨hn1, hn2⟩ := hn
    have hQ : Q (k + 1) = 5 * Q k := Q_succ k
    have hp := Q_pos k
    have hn2' : n ≤ 30 * Q k := by omega
    by_cases hle : n ≤ 6 * Q k
    · exact ih (Set.mem_Icc.mpr ⟨hn1, hle⟩)
    · push_neg at hle
      by_cases b1 : n ≤ 7 * Q k
      · exact pair_mem (2 * Q k) (3 * Q k) (4 * Q k) (4 * Q k) n (by omega) (by omega)
          (hIsub k) (haI k) (by omega) (by omega)
      · push_neg at b1
        by_cases b2 : n ≤ 9 * Q k - 1
        · exact pair_mem (2 * Q k) (3 * Q k) (5 * Q k) (6 * Q k - 1) n (by omega) (by omega)
            (hIsub k) (hBsub k) (by omega) (by omega)
        · push_neg at b2
          by_cases b3 : n ≤ 10 * Q k - 1
          · exact pair_mem (4 * Q k) (4 * Q k) (5 * Q k) (6 * Q k - 1) n (by omega) (by omega)
              (haI k) (hBsub k) (by omega) (by omega)
          · push_neg at b3
            by_cases b4 : n ≤ 12 * Q k - 2
            · exact pair_mem (5 * Q k) (6 * Q k - 1) (5 * Q k) (6 * Q k - 1) n (by omega) (by omega)
                (hBsub k) (hBsub k) (by omega) (by omega)
            · push_neg at b4
              by_cases b5 : n ≤ 18 * Q k
              · exact pair_mem (2 * Q k) (3 * Q k) (10 * Q k - 1) (15 * Q k) n (by omega) (by omega)
                  (hIsub k) (hCsub k) (by omega) (by omega)
              · push_neg at b5
                by_cases b6 : n ≤ 21 * Q k - 1
                · exact pair_mem (5 * Q k) (6 * Q k - 1) (10 * Q k - 1) (15 * Q k) n (by omega) (by omega)
                    (hBsub k) (hCsub k) (by omega) (by omega)
                · push_neg at b6
                  exact pair_mem (10 * Q k - 1) (15 * Q k) (10 * Q k - 1) (15 * Q k) n (by omega) (by omega)
                    (hCsub k) (hCsub k) (by omega) (by omega)

lemma gap_lem (T : Set ℕ) (k : ℕ) (hT : T ⊆ setA) (h4 : 4 * Q k ∉ T)
    (m : ℕ) (hm : m ∈ T + T) : m ∉ Ico (9 * Q k) (10 * Q k) := by
  intro hmJ
  rw [Set.mem_add] at hm
  obtain ⟨x, hx, y, hy, hxy⟩ := hm
  rw [Set.mem_Ico] at hmJ
  have hxA := hT hx
  have hyA := hT hy
  have hforce := sum_in_J_forces k x y hxA hyA (by omega) (by omega)
  rcases hforce with hx4 | hy4
  · exact h4 (hx4 ▸ hx)
  · exact h4 (hy4 ▸ hy)

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
    have hn6 : n ≤ 6 * Q n := by have := Q_gt n; omega
    have hmem := basis_up n (Set.mem_Icc.mpr ⟨hn, hn6⟩)
    rw [Set.mem_add] at hmem
    obtain ⟨a, haA, b, hbA, hab⟩ := hmem
    exact ⟨a, haA, b, hbA, hab⟩
  · intro A₁ A₂ hA1 hA2 hcov hdisj hsyn
    obtain ⟨hs1, hs2⟩ := hsyn
    unfold IsSyndetic at hs1 hs2
    obtain ⟨C₁, hsyn1⟩ := hs1
    obtain ⟨C₂, hsyn2⟩ := hs2
    set k := C₁ + C₂ + 1 with hkdef
    have hk1 : C₁ < Q k := by have := Q_gt k; omega
    have hk2 : C₂ < Q k := by have := Q_gt k; omega
    have h4mem : 4 * Q k ∈ setA := haI k (Set.mem_Icc.mpr ⟨le_refl _, le_refl _⟩)
    have hcov4 : 4 * Q k ∈ A₁ ∨ 4 * Q k ∈ A₂ := hcov (4 * Q k) h4mem
    rcases hcov4 with h1 | h2
    · have hnot : 4 * Q k ∉ A₂ := by
        intro hin
        have hmem : 4 * Q k ∈ A₁ ∩ A₂ := ⟨h1, hin⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmTT, hmIcc⟩ := hsyn2 (9 * Q k)
      have hgap := gap_lem A₂ k hA2 hnot m hmTT
      apply hgap
      rw [Set.mem_Ico]; rw [Set.mem_Icc] at hmIcc; omega
    · have hnot : 4 * Q k ∉ A₁ := by
        intro hin
        have hmem : 4 * Q k ∈ A₁ ∩ A₂ := ⟨hin, h2⟩
        rw [hdisj] at hmem; exact hmem
      obtain ⟨m, hmTT, hmIcc⟩ := hsyn1 (9 * Q k)
      have hgap := gap_lem A₁ k hA1 hnot m hmTT
      apply hgap
      rw [Set.mem_Ico]; rw [Set.mem_Icc] at hmIcc; omega

end Erdos741OAI

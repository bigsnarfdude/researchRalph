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
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_one_le (k : ℕ) : 1 ≤ Q k := Q_pos k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

/-! ## setA membership helpers -/

lemma two_mem_setA : (2 : ℕ) ∈ setA := Or.inl (by simp)
lemma three_mem_setA : (3 : ℕ) ∈ setA := Or.inl (by simp)

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

lemma Bk_mem_setA {k x : ℕ} (hx : x ∈ Bk k) : x ∈ setA :=
  Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inr hx)⟩)

lemma Fk_mem_setA {k x : ℕ} (hx : x ∈ Fk k) : x ∈ setA :=
  Or.inr (mem_iUnion.mpr ⟨k, Or.inr hx⟩)

/-- The inherited "I" interval [2 Qk, 3 Qk] lands in setA (via {2,3} at level 0,
    via Fk (k-1) for k ≥ 1). -/
lemma I_mem_setA : ∀ k x, x ∈ Icc (2 * Q k) (3 * Q k) → x ∈ setA
  | 0, x, hx => by
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      rcases (by omega : x = 2 ∨ x = 3) with h | h
      · exact h ▸ two_mem_setA
      · exact h ▸ three_mem_setA
  | (k+1), x, hx => by
      rw [Q_succ] at hx
      simp only [mem_Icc] at hx
      apply Fk_mem_setA (k := k)
      simp only [Fk, mem_Icc]
      have hq := Q_pos k
      omega

/-! ## Interval-sum cover helper -/

lemma interval_sum_cover {a1 a2 b1 b2 n : ℕ}
    (h1 : a1 + b1 ≤ n) (h2 : n ≤ a2 + b2) (h3 : a1 ≤ a2) (h4 : b1 ≤ b2) :
    ∃ i, a1 ≤ i ∧ i ≤ a2 ∧ ∃ j, b1 ≤ j ∧ j ≤ b2 ∧ i + j = n := by
  refine ⟨min a2 (n - b1), ?_, ?_, n - min a2 (n - b1), ?_, ?_, ?_⟩ <;> omega

/-! ## Basis: every n ≥ 4 is a sum of two elements of setA -/

lemma cover : ∀ k n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro k
  induction k with
  | zero =>
      intro n hn4 hle
      simp only [Q, pow_zero, mul_one] at hle
      interval_cases n
      · exact ⟨2, two_mem_setA, 2, two_mem_setA, rfl⟩
      · exact ⟨2, two_mem_setA, 3, three_mem_setA, rfl⟩
      · exact ⟨3, three_mem_setA, 3, three_mem_setA, rfl⟩
  | succ k ih =>
      intro n hn4 hle
      rw [Q_succ] at hle
      have hq := Q_pos k
      by_cases hsmall : n ≤ 6 * Q k
      · exact ih n hn4 hsmall
      · push_neg at hsmall
        by_cases hc1 : n ≤ 7 * Q k
        · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
            interval_sum_cover (n := n) (a1 := 2*Q k) (b1 := 4*Q k) (a2 := 3*Q k) (b2 := 4*Q k)
              (by omega) (by omega) (by omega) (by omega)
          refine ⟨i, I_mem_setA k i (mem_Icc.mpr ⟨hi1, hi2⟩), j, ?_, hs⟩
          have : j = ck k := by simp only [ck]; omega
          exact this ▸ ck_mem_setA k
        · by_cases hc2 : n ≤ 9 * Q k - 1
          · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
              interval_sum_cover (n := n) (a1 := 2*Q k) (b1 := 5*Q k) (a2 := 3*Q k) (b2 := 6*Q k - 1)
                (by omega) (by omega) (by omega) (by omega)
            exact ⟨i, I_mem_setA k i (mem_Icc.mpr ⟨hi1, hi2⟩), j,
                   Bk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩
          · by_cases hc3 : n ≤ 10 * Q k - 1
            · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
                interval_sum_cover (n := n) (a1 := 4*Q k) (b1 := 5*Q k) (a2 := 4*Q k) (b2 := 6*Q k - 1)
                  (by omega) (by omega) (by omega) (by omega)
              refine ⟨i, ?_, j, Bk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩
              have : i = ck k := by simp only [ck]; omega
              exact this ▸ ck_mem_setA k
            · by_cases hc4 : n ≤ 12 * Q k - 2
              · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
                  interval_sum_cover (n := n) (a1 := 5*Q k) (b1 := 5*Q k) (a2 := 6*Q k - 1) (b2 := 6*Q k - 1)
                    (by omega) (by omega) (by omega) (by omega)
                exact ⟨i, Bk_mem_setA (mem_Icc.mpr ⟨hi1, hi2⟩), j,
                       Bk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩
              · by_cases hc5 : n ≤ 18 * Q k
                · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
                    interval_sum_cover (n := n) (a1 := 2*Q k) (b1 := 10*Q k - 1) (a2 := 3*Q k) (b2 := 15*Q k)
                      (by omega) (by omega) (by omega) (by omega)
                  exact ⟨i, I_mem_setA k i (mem_Icc.mpr ⟨hi1, hi2⟩), j,
                         Fk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩
                · by_cases hc6 : n ≤ 21 * Q k - 1
                  · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
                      interval_sum_cover (n := n) (a1 := 5*Q k) (b1 := 10*Q k - 1) (a2 := 6*Q k - 1) (b2 := 15*Q k)
                        (by omega) (by omega) (by omega) (by omega)
                    exact ⟨i, Bk_mem_setA (mem_Icc.mpr ⟨hi1, hi2⟩), j,
                           Fk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩
                  · obtain ⟨i, hi1, hi2, j, hj1, hj2, hs⟩ :=
                      interval_sum_cover (n := n) (a1 := 10*Q k - 1) (b1 := 10*Q k - 1) (a2 := 15*Q k) (b2 := 15*Q k)
                        (by omega) (by omega) (by omega) (by omega)
                    exact ⟨i, Fk_mem_setA (mem_Icc.mpr ⟨hi1, hi2⟩), j,
                           Fk_mem_setA (mem_Icc.mpr ⟨hj1, hj2⟩), hs⟩

lemma basis_setA : ∀ n, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  apply cover n n hn
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : 2 ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  simp only [Q]
  omega

/-! ## Geometric growth/decay of Q -/

lemma Q_small {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have hjk : j + 1 ≤ k := h
  have hm := Q_mono hjk
  rw [Q_succ] at hm
  exact hm

lemma Q_big {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have hkj : k + 1 ≤ j := h
  have hm := Q_mono hkj
  rw [Q_succ] at hm
  exact hm

lemma lt_Q (n : ℕ) : n < Q n := by
  have h1 : n < 2 ^ n := Nat.lt_two_pow_self
  have h2 : 2 ^ n ≤ 5 ^ n := Nat.pow_le_pow_left (by norm_num) n
  simp only [Q]; omega

/-! ## Decomposition of setA elements -/

lemma setA_cases {x : ℕ} (hx : x ∈ setA) :
    x = 2 ∨ x = 3 ∨ ∃ j, x = ck j ∨ x ∈ Bk j ∨ x ∈ Fk j := by
  simp only [setA, mem_union, mem_iUnion, mem_singleton_iff, mem_insert_iff] at hx
  rcases hx with (h | h) | ⟨j, hj⟩
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr ⟨j, by tauto⟩)

lemma setA_dichotomy {x : ℕ} (hx : x ∈ setA) :
    (2 ≤ x ∧ x ≤ 3) ∨ ∃ j, x = ck j ∨ x ∈ Bk j ∨ x ∈ Fk j := by
  rcases setA_cases hx with h | h | h
  · exact Or.inl (by omega)
  · exact Or.inl (by omega)
  · exact Or.inr h

lemma stage_lb {x j : ℕ} (h : x = ck j ∨ x ∈ Bk j ∨ x ∈ Fk j) : 4 * Q j ≤ x := by
  have hq := Q_pos j
  rcases h with h | h | h
  · simp only [ck] at h; omega
  · simp only [Bk, mem_Icc] at h; omega
  · simp only [Fk, mem_Icc] at h; omega

lemma stage_ub {x j : ℕ} (h : x = ck j ∨ x ∈ Bk j ∨ x ∈ Fk j) : x ≤ 15 * Q j := by
  have hq := Q_pos j
  rcases h with h | h | h
  · simp only [ck] at h; omega
  · simp only [Bk, mem_Icc] at h; omega
  · simp only [Fk, mem_Icc] at h; omega

/-! ## Rigidity: sums landing in Jk k must be ck k + Bk k -/

lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hn : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hlo, hhi⟩ := hn
  have hq := Q_pos k
  rcases setA_dichotomy ha with hasmall | ⟨p, hap⟩
  · rcases setA_dichotomy hb with hbsmall | ⟨q, hbq⟩
    · exfalso; omega
    · exfalso
      have hbq_lb := stage_lb hbq
      have hbq_ub := stage_ub hbq
      have hqp := Q_pos q
      rcases lt_trichotomy q k with hqlt | hqe | hqgt
      · have hqs := Q_small hqlt; omega
      · rw [hqe] at hbq
        rcases hbq with hb' | hb' | hb'
        · simp only [ck] at hb'; omega
        · simp only [Bk, mem_Icc] at hb'; omega
        · simp only [Fk, mem_Icc] at hb'; omega
      · have hqg := Q_big hqgt; omega
  · rcases setA_dichotomy hb with hbsmall | ⟨q, hbq⟩
    · exfalso
      have hap_lb := stage_lb hap
      have hap_ub := stage_ub hap
      have hpp := Q_pos p
      rcases lt_trichotomy p k with hplt | hpe | hpgt
      · have hps := Q_small hplt; omega
      · rw [hpe] at hap
        rcases hap with ha' | ha' | ha'
        · simp only [ck] at ha'; omega
        · simp only [Bk, mem_Icc] at ha'; omega
        · simp only [Fk, mem_Icc] at ha'; omega
      · have hpg := Q_big hpgt; omega
    · have hap_lb := stage_lb hap
      have hap_ub := stage_ub hap
      have hbq_lb := stage_lb hbq
      have hbq_ub := stage_ub hbq
      have hpp := Q_pos p
      have hqp := Q_pos q
      have haa : a < 10 * Q k := by omega
      have hbb : b < 10 * Q k := by omega
      rcases lt_trichotomy p k with hplt | hpe | hpgt
      · have hps := Q_small hplt
        rcases lt_trichotomy q k with hqlt | hqe | hqgt
        · have hqs := Q_small hqlt; exfalso; omega
        · rw [hqe] at hbq
          exfalso
          rcases hbq with hb' | hb' | hb'
          · simp only [ck] at hb'; omega
          · simp only [Bk, mem_Icc] at hb'; omega
          · simp only [Fk, mem_Icc] at hb'; omega
        · have hqg := Q_big hqgt; exfalso; omega
      · rcases lt_trichotomy q k with hqlt | hqe | hqgt
        · have hqs := Q_small hqlt
          rw [hpe] at hap
          exfalso
          rcases hap with ha' | ha' | ha'
          · simp only [ck] at ha'; omega
          · simp only [Bk, mem_Icc] at ha'; omega
          · simp only [Fk, mem_Icc] at ha'; omega
        · rw [hpe] at hap
          rw [hqe] at hbq
          rcases hap with ha' | ha' | ha'
          · rcases hbq with hb' | hb' | hb'
            · exfalso; simp only [ck] at ha' hb'; omega
            · exact Or.inl ⟨ha', hb'⟩
            · exfalso; simp only [ck] at ha'; simp only [Fk, mem_Icc] at hb'; omega
          · rcases hbq with hb' | hb' | hb'
            · exact Or.inr ⟨hb', ha'⟩
            · exfalso; simp only [Bk, mem_Icc] at ha' hb'; omega
            · exfalso; simp only [Bk, mem_Icc] at ha'; simp only [Fk, mem_Icc] at hb'; omega
          · rcases hbq with hb' | hb' | hb'
            · exfalso; simp only [ck] at hb'; simp only [Fk, mem_Icc] at ha'; omega
            · exfalso; simp only [Fk, mem_Icc] at ha'; simp only [Bk, mem_Icc] at hb'; omega
            · exfalso; simp only [Fk, mem_Icc] at ha' hb'; omega
        · have hqg := Q_big hqgt; exfalso; omega
      · have hpg := Q_big hpgt; exfalso; omega

/-! ## Gap lemma -/

lemma gap_lem {T : Set ℕ} (hT : T ⊆ setA) {k : ℕ} (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hnJ, hnTT⟩
  rw [Set.mem_add] at hnTT
  obtain ⟨a, haT, b, hbT, hab⟩ := hnTT
  have hjk : a + b ∈ Jk k := by rw [hab]; exact hnJ
  rcases rigidity (hT haT) (hT hbT) hjk with ⟨hc, _⟩ | ⟨hc, _⟩
  · exact hck (hc ▸ haT)
  · exact hck (hc ▸ hbT)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, basis_setA, ?_⟩
  intro A₁ A₂ h1 h2 hcov hdisj
  rintro ⟨⟨C1, hsyn1⟩, ⟨C2, hsyn2⟩⟩
  set k := C1 + C2 + 1 with hkdef
  have hQk : k < Q k := lt_Q k
  have hC1 : C1 < Q k := by omega
  have hC2 : C2 < Q k := by omega
  rcases hcov (ck k) (ck_mem_setA k) with hck1 | hck2
  · have hnot : ck k ∉ A₂ := by
      intro hmem
      have hx : ck k ∈ A₁ ∩ A₂ := ⟨hck1, hmem⟩
      rw [hdisj] at hx; exact hx
    have hgap := gap_lem h2 hnot
    obtain ⟨m, hmS, hmIcc⟩ := hsyn2 (9 * Q k)
    simp only [mem_Icc] at hmIcc
    have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
    have hcontr : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
    rw [hgap] at hcontr; exact hcontr
  · have hnot : ck k ∉ A₁ := by
      intro hmem
      have hx : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hck2⟩
      rw [hdisj] at hx; exact hx
    have hgap := gap_lem h1 hnot
    obtain ⟨m, hmS, hmIcc⟩ := hsyn1 (9 * Q k)
    simp only [mem_Icc] at hmIcc
    have hmJ : m ∈ Jk k := by simp only [Jk, mem_Ico]; omega
    have hcontr : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
    rw [hgap] at hcontr; exact hcontr

end Erdos741OAI

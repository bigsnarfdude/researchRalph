import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

/-! ## Construction -/

def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

/-- Partial union through level k. -/
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ stage k

/-! ## Basic arithmetic on Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma one_le_Q (k : ℕ) : 1 ≤ Q k := Q_pos k

/-! ## Akn monotonicity and inclusion in setA -/

lemma stage_subset_setA (k : ℕ) : stage k ⊆ setA := by
  intro x hx
  exact Or.inr (Set.mem_iUnion.mpr ⟨k, hx⟩)

lemma Akn_mono : ∀ k, Akn k ⊆ Akn (k+1) := by
  intro k x hx
  exact Or.inl hx

lemma Akn_subset_setA : ∀ k, Akn k ⊆ setA := by
  intro k
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact stage_subset_setA k hx

lemma Akn_mono_le {m n : ℕ} (h : m ≤ n) : Akn m ⊆ Akn n := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans (Akn_mono _)

lemma n_le_Q : ∀ n, n ≤ Q n := by
  intro n
  induction n with
  | zero => simp [Q]
  | succ k ih =>
    have hq := Q_pos k
    rw [Q_succ]; omega

/-! ## Element membership helpers -/

lemma base2 (m : ℕ) : (2:ℕ) ∈ Akn m :=
  Akn_mono_le (Nat.zero_le m) (by simp [Akn])

lemma base3 (m : ℕ) : (3:ℕ) ∈ Akn m :=
  Akn_mono_le (Nat.zero_le m) (by simp [Akn])

lemma ck_mem (k : ℕ) : 4 * Q k ∈ Akn (k+1) := by
  refine Set.mem_union_right _ ?_
  refine Set.mem_union_left _ ?_
  refine Set.mem_union_left _ ?_
  rfl

lemma Bk_mem {k x : ℕ} (h : x ∈ Icc (5 * Q k) (6 * Q k - 1)) : x ∈ Akn (k+1) := by
  refine Set.mem_union_right _ ?_
  refine Set.mem_union_left _ ?_
  refine Set.mem_union_right _ ?_
  exact h

lemma Fk_mem {k x : ℕ} (h : x ∈ Icc (10 * Q k - 1) (15 * Q k)) : x ∈ Akn (k+1) := by
  refine Set.mem_union_right _ ?_
  refine Set.mem_union_right _ ?_
  exact h

lemma I_mem {k x : ℕ} (h : x ∈ Icc (2 * Q k) (3 * Q k)) : x ∈ Akn (k+1) := by
  rw [mem_Icc] at h
  cases k with
  | zero =>
    simp only [Q, pow_zero, mul_one] at h
    have hx : x = 2 ∨ x = 3 := by omega
    rcases hx with rfl | rfl
    · exact base2 1
    · exact base3 1
  | succ j =>
    have hQ : Q (j+1) = 5 * Q j := Q_succ j
    have hxF : x ∈ Icc (10 * Q j - 1) (15 * Q j) := by
      rw [mem_Icc]
      rw [hQ] at h
      omega
    exact Akn_mono (j+1) (Fk_mem hxF)

/-! ## Basis lemma -/

lemma basis_lem : ∀ k, Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  intro k
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    obtain ⟨hx1, hx2⟩ := hx
    rw [Set.mem_add]
    have h2 := base2 1
    have h3 := base3 1
    interval_cases x
    · exact ⟨2, h2, 2, h2, rfl⟩
    · exact ⟨2, h2, 3, h3, rfl⟩
    · exact ⟨3, h3, 3, h3, rfl⟩
  | succ k ih =>
    intro x hx
    rw [mem_Icc] at hx
    have hpos : 0 < Q k := Q_pos k
    have hQ : Q (k+1) = 5 * Q k := Q_succ k
    have lift : Akn (k+1) + Akn (k+1) ⊆ Akn (k+2) + Akn (k+2) :=
      Set.add_subset_add (Akn_mono (k+1)) (Akn_mono (k+1))
    by_cases hxle : x ≤ 6 * Q k
    · exact lift (ih (mem_Icc.mpr ⟨hx.1, hxle⟩))
    · push_neg at hxle
      apply lift
      rw [Set.mem_add]
      rw [hQ] at hx
      by_cases h7 : x ≤ 7 * Q k
      · exact ⟨x - 4 * Q k, I_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
              4 * Q k, ck_mem k, by omega⟩
      · push_neg at h7
        by_cases h9 : x ≤ 9 * Q k - 1
        · exact ⟨max (2 * Q k) (x - (6 * Q k - 1)),
                  I_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
                  x - max (2 * Q k) (x - (6 * Q k - 1)),
                  Bk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
        · push_neg at h9
          by_cases h10 : x ≤ 10 * Q k - 1
          · exact ⟨4 * Q k, ck_mem k, x - 4 * Q k,
                    Bk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
          · push_neg at h10
            by_cases h12 : x ≤ 12 * Q k - 2
            · exact ⟨max (5 * Q k) (x - (6 * Q k - 1)),
                      Bk_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
                      x - max (5 * Q k) (x - (6 * Q k - 1)),
                      Bk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
            · push_neg at h12
              by_cases h18 : x ≤ 18 * Q k
              · exact ⟨max (2 * Q k) (x - 15 * Q k),
                        I_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
                        x - max (2 * Q k) (x - 15 * Q k),
                        Fk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
              · push_neg at h18
                by_cases h21 : x ≤ 21 * Q k - 1
                · exact ⟨max (5 * Q k) (x - 15 * Q k),
                          Bk_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
                          x - max (5 * Q k) (x - 15 * Q k),
                          Fk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
                · push_neg at h21
                  exact ⟨max (10 * Q k - 1) (x - 15 * Q k),
                          Fk_mem (mem_Icc.mpr ⟨by omega, by omega⟩),
                          x - max (10 * Q k - 1) (x - 15 * Q k),
                          Fk_mem (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩

/-! ## Rigidity machinery -/

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num) h

lemma stage_lb {j x : ℕ} (h : x ∈ stage j) : 4 * Q j ≤ x := by
  have hp := Q_pos j
  simp only [stage, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

lemma stage_ub {j x : ℕ} (h : x ∈ stage j) : x ≤ 15 * Q j := by
  have hp := Q_pos j
  simp only [stage, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at h
  rcases h with (h | h) | h <;> omega

lemma stage_small {j k x : ℕ} (hjk : j < k) (hx : x ∈ stage j) : x ≤ 3 * Q k := by
  have hub := stage_ub hx
  have hle : Q (j+1) ≤ Q k := Q_mono (Nat.succ_le_of_lt hjk)
  have h5 : Q (j+1) = 5 * Q j := Q_succ j
  omega

lemma idx_le {j k : ℕ} (h : 4 * Q j < 10 * Q k) : j ≤ k := by
  by_contra hlt
  push_neg at hlt
  have hle : Q (k+1) ≤ Q j := Q_mono (Nat.succ_le_of_lt hlt)
  have h5 : Q (k+1) = 5 * Q k := Q_succ k
  have hp := Q_pos k
  omega

lemma mem_stage_iff {j x : ℕ} : x ∈ stage j ↔
    x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨ (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  constructor
  · intro h
    simp only [stage, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at h
    tauto
  · intro h
    simp only [stage, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc]
    tauto

lemma setA_elt {k e : ℕ} (he : e ∈ setA) (hlt : e < 10 * Q k) :
    (e = 2 ∨ e = 3) ∨ ∃ j, j ≤ k ∧ e ∈ stage j := by
  simp only [setA, Set.mem_union, Set.mem_iUnion, Set.mem_insert_iff,
             Set.mem_singleton_iff] at he
  rcases he with h23 | ⟨j, hj⟩
  · exact Or.inl h23
  · refine Or.inr ⟨j, ?_, hj⟩
    have hlb := stage_lb hj
    exact idx_le (by omega)

lemma rigidity {k n a b : ℕ} (hn : n ∈ Jk k)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    a = 4 * Q k ∨ b = 4 * Q k := by
  have hpos : 0 < Q k := Q_pos k
  rw [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have ha10 : a < 10 * Q k := by omega
  have hb10 : b < 10 * Q k := by omega
  rcases setA_elt ha ha10 with ha23 | ⟨ja, hjak, hjas⟩
  · rcases setA_elt hb hb10 with hb23 | ⟨jb, hjbk, hjbs⟩
    · exfalso; rcases ha23 with rfl | rfl <;> rcases hb23 with rfl | rfl <;> omega
    · exfalso
      have hpjb := Q_pos jb
      rcases hjbk.lt_or_eq with hlt | heq
      · have hbsm := stage_small hlt hjbs
        rcases ha23 with rfl | rfl <;> omega
      · subst heq
        rw [mem_stage_iff] at hjbs
        rcases ha23 with rfl | rfl <;> rcases hjbs with h | h | h <;> omega
  · rcases setA_elt hb hb10 with hb23 | ⟨jb, hjbk, hjbs⟩
    · exfalso
      have hpja := Q_pos ja
      rcases hjak.lt_or_eq with hlt | heq
      · have hasm := stage_small hlt hjas
        rcases hb23 with rfl | rfl <;> omega
      · subst heq
        rw [mem_stage_iff] at hjas
        rcases hb23 with rfl | rfl <;> rcases hjas with h | h | h <;> omega
    · have hpja := Q_pos ja
      have hpjb := Q_pos jb
      rcases hjak.lt_or_eq with hjalt | hjaeq
      · rcases hjbk.lt_or_eq with hjblt | hjbeq
        · exfalso
          have hax := stage_small hjalt hjas
          have hbx := stage_small hjblt hjbs
          omega
        · exfalso
          subst hjbeq
          have hasm := stage_small hjalt hjas
          have halb := stage_lb hjas
          rw [mem_stage_iff] at hjbs
          rcases hjbs with h | h | h <;> omega
      · rcases hjbk.lt_or_eq with hjblt | hjbeq
        · exfalso
          subst hjaeq
          have hbsm := stage_small hjblt hjbs
          have hblb := stage_lb hjbs
          rw [mem_stage_iff] at hjas
          rcases hjas with h | h | h <;> omega
        · subst hjaeq; subst hjbeq
          rw [mem_stage_iff] at hjas hjbs
          rcases hjas with h | h | h <;> rcases hjbs with h2 | h2 | h2 <;>
            first
              | (left; omega)
              | (right; omega)
              | (exfalso; omega)

lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : 4 * Q k ∉ T)
    {n : ℕ} (hn : n ∈ Jk k) (hmem : n ∈ T + T) : False := by
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  rcases rigidity hn (hT ha) (hT hb) hab with h | h
  · exact hck (h ▸ ha)
  · exact hck (h ▸ hb)

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
    have hnle : n ≤ 6 * Q n := le_trans (n_le_Q n) (by omega)
    have hmem : n ∈ Icc 4 (6 * Q n) := mem_Icc.mpr ⟨hn, hnle⟩
    have hsum := basis_lem n hmem
    rw [Set.mem_add] at hsum
    obtain ⟨a, ha, b, hb, hab⟩ := hsum
    exact ⟨a, Akn_subset_setA _ ha, b, Akn_subset_setA _ hb, hab⟩
  · intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    set k : ℕ := C₁ + C₂ + 1 with hk
    have hkQ : k ≤ Q k := n_le_Q k
    have hpos : 0 < Q k := Q_pos k
    have hck_mem : 4 * Q k ∈ setA := Akn_subset_setA (k+1) (ck_mem k)
    rcases hcov _ hck_mem with hin1 | hin2
    · have hck2 : 4 * Q k ∉ A₂ := by
        intro hc
        have h : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hin1, hc⟩
        rw [hdisj, Set.mem_empty_iff_false] at h
        exact h
      obtain ⟨m, hmS, hmI⟩ := hC₂ (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]
        exact ⟨hmI.1, by omega⟩
      exact gap_lem h2 hck2 hmJ hmS
    · have hck1 : 4 * Q k ∉ A₁ := by
        intro hc
        have h : (4 * Q k) ∈ A₁ ∩ A₂ := ⟨hc, hin2⟩
        rw [hdisj, Set.mem_empty_iff_false] at h
        exact h
      obtain ⟨m, hmS, hmI⟩ := hC₁ (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]
        exact ⟨hmI.1, by omega⟩
      exact gap_lem h1 hck1 hmJ hmS

end Erdos741OAI

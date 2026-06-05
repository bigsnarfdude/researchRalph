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

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ stage k

/-! ## Arithmetic on Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_le {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

lemma Q_lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have h2 : Q (j+1) ≤ Q k := Q_le h
  rw [Q_succ] at h2
  exact h2

/-! ## Monotonicity of the partial unions -/

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  exact Or.inl hx

lemma akn_mono_le {j k : ℕ} (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans (akn_mono _)

lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
      intro x hx
      rcases hx with hx | hx
      · exact ih hx
      · exact Or.inr (mem_iUnion.mpr ⟨k, hx⟩)

lemma stage_sub_akn (k : ℕ) : stage k ⊆ Akn (k+1) := by
  intro x hx
  exact Or.inr hx

/-! ## The I-interval is available one level up -/

lemma I_mem (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+1) := by
  cases k with
  | zero =>
      intro x hx
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      show x ∈ Akn 0 ∪ stage 0
      refine Or.inl ?_
      show x ∈ ({2,3} : Set ℕ)
      rcases (by omega : x = 2 ∨ x = 3) with rfl | rfl <;> simp
  | succ j =>
      intro x hx
      have hFk : x ∈ Fk j := by
        simp only [Fk, mem_Icc]
        rw [mem_Icc, Q_succ] at hx
        omega
      have hx1 : x ∈ Akn (j+1) := Or.inr (Or.inr hFk)
      exact akn_mono (j+1) hx1

/-! ## A is a basis -/

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
      intro x hx
      simp only [Q, pow_zero, mul_one, mem_Icc] at hx
      have h2 : (2:ℕ) ∈ Akn 1 :=
        akn_mono_le (Nat.zero_le 1) (by show (2:ℕ) ∈ ({2,3}:Set ℕ); simp)
      have h3 : (3:ℕ) ∈ Akn 1 :=
        akn_mono_le (Nat.zero_le 1) (by show (3:ℕ) ∈ ({2,3}:Set ℕ); simp)
      rcases (by omega : x = 4 ∨ x = 5 ∨ x = 6) with rfl | rfl | rfl
      · exact Set.mem_add.mpr ⟨2, h2, 2, h2, rfl⟩
      · exact Set.mem_add.mpr ⟨2, h2, 3, h3, rfl⟩
      · exact Set.mem_add.mpr ⟨3, h3, 3, h3, rfl⟩
  | succ k ih =>
      intro x hx
      rw [mem_Icc] at hx
      obtain ⟨hx4, hxhi⟩ := hx
      rw [Q_succ] at hxhi
      have hqp : 0 < Q k := Q_pos k
      have hIsub : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k+2) :=
        fun y hy => akn_mono (k+1) (I_mem k hy)
      have hck : ck k ∈ Akn (k+2) :=
        akn_mono (k+1) (stage_sub_akn k (Or.inl (Or.inl rfl)))
      have hBsub : Icc (5 * Q k) (6 * Q k - 1) ⊆ Akn (k+2) :=
        fun y hy => akn_mono (k+1) (stage_sub_akn k (Or.inl (Or.inr hy)))
      have hFsub : Icc (10 * Q k - 1) (15 * Q k) ⊆ Akn (k+2) :=
        fun y hy => akn_mono (k+1) (stage_sub_akn k (Or.inr hy))
      by_cases hsmall : x ≤ 6 * Q k
      · -- small region: covered by induction hypothesis
        have hmem := ih (mem_Icc.mpr ⟨hx4, hsmall⟩)
        obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmem
        exact Set.mem_add.mpr ⟨a, akn_mono _ ha, b, akn_mono _ hb, hab⟩
      · push_neg at hsmall
        by_cases c2 : x ≤ 7 * Q k
        · -- I + ck
          exact Set.mem_add.mpr
            ⟨x - 4 * Q k, hIsub (mem_Icc.mpr ⟨by omega, by omega⟩),
             4 * Q k, hck, by omega⟩
        · push_neg at c2
          by_cases c3 : x ≤ 9 * Q k - 1
          · -- I + Bk
            exact Set.mem_add.mpr
              ⟨max (2 * Q k) (x - (6 * Q k - 1)),
               hIsub (mem_Icc.mpr ⟨by omega, by omega⟩),
               x - max (2 * Q k) (x - (6 * Q k - 1)),
               hBsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
          · push_neg at c3
            by_cases c4 : x ≤ 10 * Q k - 1
            · -- ck + Bk
              exact Set.mem_add.mpr
                ⟨4 * Q k, hck, x - 4 * Q k,
                 hBsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
            · push_neg at c4
              by_cases c5 : x ≤ 12 * Q k - 2
              · -- Bk + Bk
                exact Set.mem_add.mpr
                  ⟨max (5 * Q k) (x - (6 * Q k - 1)),
                   hBsub (mem_Icc.mpr ⟨by omega, by omega⟩),
                   x - max (5 * Q k) (x - (6 * Q k - 1)),
                   hBsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
              · push_neg at c5
                by_cases c6 : x ≤ 18 * Q k
                · -- I + Fk
                  exact Set.mem_add.mpr
                    ⟨max (2 * Q k) (x - 15 * Q k),
                     hIsub (mem_Icc.mpr ⟨by omega, by omega⟩),
                     x - max (2 * Q k) (x - 15 * Q k),
                     hFsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
                · push_neg at c6
                  by_cases c7 : x ≤ 21 * Q k - 1
                  · -- Bk + Fk
                    exact Set.mem_add.mpr
                      ⟨max (5 * Q k) (x - 15 * Q k),
                       hBsub (mem_Icc.mpr ⟨by omega, by omega⟩),
                       x - max (5 * Q k) (x - 15 * Q k),
                       hFsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩
                  · push_neg at c7
                    -- Fk + Fk
                    exact Set.mem_add.mpr
                      ⟨max (10 * Q k - 1) (x - 15 * Q k),
                       hFsub (mem_Icc.mpr ⟨by omega, by omega⟩),
                       x - max (10 * Q k - 1) (x - 15 * Q k),
                       hFsub (mem_Icc.mpr ⟨by omega, by omega⟩), by omega⟩

lemma setA_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  have hlt : n < Q n := by
    have : n < 5 ^ n := Nat.lt_pow_self (by norm_num)
    exact this
  have hk : n ≤ 6 * Q n := by omega
  obtain ⟨a, ha, b, hb, hab⟩ :=
    Set.mem_add.mp (basis_lem n (mem_Icc.mpr ⟨hn, hk⟩))
  exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩

/-! ## Rigidity around the gap zone -/

lemma setA_desc {e : ℕ} (he : e ∈ setA) :
    e = 2 ∨ e = 3 ∨ ∃ j, e = 4 * Q j ∨ (5 * Q j ≤ e ∧ e ≤ 6 * Q j - 1)
      ∨ (10 * Q j - 1 ≤ e ∧ e ≤ 15 * Q j) := by
  rcases he with he | he
  · simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at he
    rcases he with rfl | rfl
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)
  · rw [mem_iUnion] at he
    obtain ⟨j, hj⟩ := he
    refine Or.inr (Or.inr ⟨j, ?_⟩)
    simp only [stage, Set.mem_union] at hj
    rcases hj with (hj | hj) | hj
    · rw [Set.mem_singleton_iff] at hj
      exact Or.inl hj
    · simp only [Bk, mem_Icc] at hj
      exact Or.inr (Or.inl hj)
    · simp only [Fk, mem_Icc] at hj
      exact Or.inr (Or.inr hj)

lemma two_le {e : ℕ} (he : e ∈ setA) : 2 ≤ e := by
  rcases setA_desc he with rfl | rfl | ⟨j, hj⟩
  · omega
  · omega
  · have := Q_pos j
    rcases hj with h | h | h <;> omega

lemma locate (k : ℕ) {e : ℕ} (he : e ∈ setA) (hlt : e < 10 * Q k) :
    e ≤ 3 * Q k ∨ e = 4 * Q k ∨ (5 * Q k ≤ e ∧ e ≤ 6 * Q k - 1) ∨ e = 10 * Q k - 1 := by
  have hkp := Q_pos k
  rcases setA_desc he with rfl | rfl | ⟨j, hj⟩
  · left; omega
  · left; omega
  · have hjp := Q_pos j
    rcases lt_trichotomy j k with hjk | hjk | hjk
    · left
      have hb : 5 * Q j ≤ Q k := Q_lt hjk
      rcases hj with h | h | h <;> omega
    · subst hjk
      rcases hj with h | h | h
      · right; left; exact h
      · right; right; left; exact h
      · right; right; right; omega
    · exfalso
      have hb : 5 * Q k ≤ Q j := Q_lt hjk
      rcases hj with h | h | h <;> omega

lemma rigidity (k n : ℕ) (hn : n ∈ Jk k) :
    ∀ a ∈ setA, ∀ b ∈ setA, a + b = n →
      (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro a ha b hb hab
  simp only [Jk, mem_Ico] at hn
  obtain ⟨h9, h10⟩ := hn
  have hkp := Q_pos k
  have h2a := two_le ha
  have h2b := two_le hb
  have hbla : a < 10 * Q k := by omega
  have hblb : b < 10 * Q k := by omega
  have la := locate k ha hbla
  have lb := locate k hb hblb
  rcases la with la | la | la | la <;>
    rcases lb with lb | lb | lb | lb <;>
    first
      | exact Or.inl ⟨la, mem_Icc.mpr lb⟩
      | exact Or.inr ⟨lb, mem_Icc.mpr la⟩
      | (exfalso; omega)

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n ∈ Jk k, n ∉ T + T := by
  intro n hn hmem
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hmem
  rcases rigidity k n hn a (hT ha) b (hT hb) hab with ⟨h1, _⟩ | ⟨h1, _⟩
  · exact hck (h1 ▸ ha)
  · exact hck (h1 ▸ hb)

/-! ## Main theorem -/

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, setA_basis, ?_⟩
  intro A1 A2 hA1 hA2 hcover hdisj
  rintro ⟨⟨C1, hC1⟩, ⟨C2, hC2⟩⟩
  set k := C1 + C2 + 1 with hkdef
  have hlt : k < Q k := by
    have : k < 5 ^ k := Nat.lt_pow_self (by norm_num)
    exact this
  have hck_memA : ck k ∈ setA :=
    Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)
  rcases hcover (ck k) hck_memA with hin1 | hin2
  · have hnotA2 : ck k ∉ A2 := fun hc =>
      (Set.mem_empty_iff_false (ck k)).mp (hdisj ▸ ⟨hin1, hc⟩)
    obtain ⟨m, hm, hmI⟩ := hC2 (9 * Q k)
    have hmJ : m ∈ Jk k := by
      simp only [mem_Icc] at hmI
      refine mem_Ico.mpr ⟨hmI.1, ?_⟩
      omega
    exact gap_lem k A2 hA2 hnotA2 m hmJ hm
  · have hnotA1 : ck k ∉ A1 := fun hc =>
      (Set.mem_empty_iff_false (ck k)).mp (hdisj ▸ ⟨hc, hin2⟩)
    obtain ⟨m, hm, hmI⟩ := hC1 (9 * Q k)
    have hmJ : m ∈ Jk k := by
      simp only [mem_Icc] at hmI
      refine mem_Ico.mpr ⟨hmI.1, ?_⟩
      omega
    exact gap_lem k A1 hA1 hnotA1 m hmJ hm

end Erdos741OAI

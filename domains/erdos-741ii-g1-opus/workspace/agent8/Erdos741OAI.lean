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
def stage (k : ℕ) : Set ℕ := ({ck k} : Set ℕ) ∪ Bk k ∪ Fk k
def setA : Set ℕ := ({2, 3} : Set ℕ) ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

/-! ## Basic facts about Q -/

lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_one_le (k : ℕ) : 1 ≤ Q k := Q_pos k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ m ih =>
    rw [Q_succ]
    have := Q_one_le m
    omega

/-! ## Membership in setA -/

lemma two_mem : (2:ℕ) ∈ setA := by
  apply mem_union_left; simp

lemma three_mem : (3:ℕ) ∈ setA := by
  apply mem_union_left; simp

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  apply mem_union_right
  apply mem_iUnion.mpr
  exact ⟨k, Or.inl (Or.inl rfl)⟩

lemma Bk_sub_setA (k : ℕ) : Bk k ⊆ setA := by
  intro x hx
  apply mem_union_right
  apply mem_iUnion.mpr
  exact ⟨k, Or.inl (Or.inr hx)⟩

lemma Fk_sub_setA (k : ℕ) : Fk k ⊆ setA := by
  intro x hx
  apply mem_union_right
  apply mem_iUnion.mpr
  exact ⟨k, Or.inr hx⟩

lemma I_sub_setA (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ setA := by
  cases k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    obtain ⟨hx1, hx2⟩ := hx
    interval_cases x
    · exact two_mem
    · exact three_mem
  | succ j =>
    intro x hx
    apply Fk_sub_setA j
    show x ∈ Fk j
    simp only [Fk, mem_Icc, mem_Icc] at *
    have hs : Q (j+1) = 5 * Q j := Q_succ j
    have hp : 1 ≤ Q j := Q_one_le j
    omega

lemma mem_I (k u : ℕ) (h1 : 2*Q k ≤ u) (h2 : u ≤ 3*Q k) : u ∈ setA :=
  I_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_B (k u : ℕ) (h1 : 5*Q k ≤ u) (h2 : u ≤ 6*Q k - 1) : u ∈ setA :=
  Bk_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_F (k u : ℕ) (h1 : 10*Q k - 1 ≤ u) (h2 : u ≤ 15*Q k) : u ∈ setA :=
  Fk_sub_setA k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_ck' (k v : ℕ) (h1 : 4*Q k ≤ v) (h2 : v ≤ 4*Q k) : v ∈ setA := by
  have : v = 4 * Q k := le_antisymm h2 h1
  rw [this]; exact ck_mem_setA k

/-! ## Interval sum helper -/

lemma pair_mem {a b c d x : ℕ} (hab : a ≤ b) (hcd : c ≤ d)
    (hlo : a + c ≤ x) (hhi : x ≤ b + d) :
    ∃ u, (a ≤ u ∧ u ≤ b) ∧ ∃ v, (c ≤ v ∧ v ≤ d) ∧ u + v = x := by
  by_cases h : x ≤ b + c
  · exact ⟨x - c, ⟨by omega, by omega⟩, c, ⟨by omega, by omega⟩, by omega⟩
  · exact ⟨b, ⟨by omega, by omega⟩, x - b, ⟨by omega, by omega⟩, by omega⟩

/-! ## Basis lemma: every n in [4, 6 Q k] is a sum of two elements of setA -/

lemma basis_lem (k : ℕ) :
    ∀ n, 4 ≤ n → n ≤ 6 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  induction k with
  | zero =>
    intro n h4 hn
    simp only [Q, pow_zero, mul_one] at hn
    obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
      pair_mem (show 2 ≤ 3 by norm_num) (show 2 ≤ 3 by norm_num)
        (show 2+2 ≤ n by omega) (show n ≤ 3+3 by omega)
    refine ⟨u, ?_, v, ?_, huv⟩
    · exact mem_I 0 u (by simpa [Q] using hu1) (by simpa [Q] using hu2)
    · exact mem_I 0 v (by simpa [Q] using hv1) (by simpa [Q] using hv2)
  | succ k ih =>
    intro n h4 hn
    have hp : 1 ≤ Q k := Q_one_le k
    have hs : Q (k+1) = 5 * Q k := Q_succ k
    rw [hs] at hn
    by_cases hle : n ≤ 6 * Q k
    · exact ih n h4 hle
    by_cases h7 : n ≤ 7 * Q k
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 2*Q k ≤ 3*Q k by omega) (show 4*Q k ≤ 4*Q k by omega)
          (show 2*Q k + 4*Q k ≤ n by omega) (show n ≤ 3*Q k + 4*Q k by omega)
      exact ⟨u, mem_I k u hu1 hu2, v, mem_ck' k v hv1 hv2, huv⟩
    by_cases h9 : n ≤ 9 * Q k - 1
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 2*Q k ≤ 3*Q k by omega) (show 5*Q k ≤ 6*Q k -1 by omega)
          (show 2*Q k + 5*Q k ≤ n by omega) (show n ≤ 3*Q k + (6*Q k -1) by omega)
      exact ⟨u, mem_I k u hu1 hu2, v, mem_B k v hv1 hv2, huv⟩
    by_cases h10 : n ≤ 10 * Q k - 1
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 4*Q k ≤ 4*Q k by omega) (show 5*Q k ≤ 6*Q k -1 by omega)
          (show 4*Q k + 5*Q k ≤ n by omega) (show n ≤ 4*Q k + (6*Q k -1) by omega)
      exact ⟨u, mem_ck' k u hu1 hu2, v, mem_B k v hv1 hv2, huv⟩
    by_cases h12 : n ≤ 12 * Q k - 2
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 5*Q k ≤ 6*Q k -1 by omega) (show 5*Q k ≤ 6*Q k -1 by omega)
          (show 5*Q k + 5*Q k ≤ n by omega) (show n ≤ (6*Q k -1) + (6*Q k -1) by omega)
      exact ⟨u, mem_B k u hu1 hu2, v, mem_B k v hv1 hv2, huv⟩
    by_cases h18 : n ≤ 18 * Q k
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 2*Q k ≤ 3*Q k by omega) (show 10*Q k -1 ≤ 15*Q k by omega)
          (show 2*Q k + (10*Q k -1) ≤ n by omega) (show n ≤ 3*Q k + 15*Q k by omega)
      exact ⟨u, mem_I k u hu1 hu2, v, mem_F k v hv1 hv2, huv⟩
    by_cases h21 : n ≤ 21 * Q k - 1
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 5*Q k ≤ 6*Q k -1 by omega) (show 10*Q k -1 ≤ 15*Q k by omega)
          (show 5*Q k + (10*Q k -1) ≤ n by omega) (show n ≤ (6*Q k -1) + 15*Q k by omega)
      exact ⟨u, mem_B k u hu1 hu2, v, mem_F k v hv1 hv2, huv⟩
    · obtain ⟨u, ⟨hu1,hu2⟩, v, ⟨hv1,hv2⟩, huv⟩ :=
        pair_mem (show 10*Q k -1 ≤ 15*Q k by omega) (show 10*Q k -1 ≤ 15*Q k by omega)
          (show (10*Q k -1) + (10*Q k -1) ≤ n by omega) (show n ≤ 15*Q k + 15*Q k by omega)
      exact ⟨u, mem_F k u hu1 hu2, v, mem_F k v hv1 hv2, huv⟩

/-! ## Rigidity machinery -/

lemma Q_mono {m n : ℕ} (h : m ≤ n) : Q m ≤ Q n :=
  Nat.pow_le_pow_right (by norm_num) h

lemma stage_lb {x j : ℕ} (hx : x ∈ stage j) : 4 * Q j ≤ x := by
  have hp := Q_one_le j
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hx
  rcases hx with ((rfl | ⟨h, _⟩) | ⟨h, _⟩) <;> omega

lemma stage_ub {x j : ℕ} (hx : x ∈ stage j) : x ≤ 15 * Q j := by
  have hp := Q_one_le j
  simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hx
  rcases hx with ((rfl | ⟨_, h⟩) | ⟨_, h⟩) <;> omega

lemma two_le_of_mem {x : ℕ} (hx : x ∈ setA) : 2 ≤ x := by
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff] at hx
  rcases hx with (h23 | ⟨j, hj⟩)
  · rcases h23 with rfl | rfl <;> omega
  · have := stage_lb hj
    have := Q_one_le j
    omega

lemma classify {x k : ℕ} (hx : x ∈ setA) (hlt : x < 10 * Q k) :
    (x ≤ 3 * Q k) ∨ (x = 4 * Q k) ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨
      (10 * Q k - 1 ≤ x ∧ x ≤ 15 * Q k) := by
  have hp : 1 ≤ Q k := Q_one_le k
  simp only [setA, mem_union, mem_iUnion, mem_insert_iff, mem_singleton_iff] at hx
  rcases hx with (h23 | ⟨j, hj⟩)
  · left; rcases h23 with rfl | rfl <;> omega
  · rcases lt_trichotomy j k with hjk | hjk | hjk
    · left
      have hub := stage_ub hj
      have h5 : 5 * Q j ≤ Q k := by
        have := Q_mono (show j + 1 ≤ k by omega); rw [Q_succ] at this; exact this
      omega
    · rw [hjk] at hj
      simp only [stage, ck, Bk, Fk, mem_union, mem_singleton_iff, mem_Icc] at hj
      rcases hj with ((h | h) | h)
      · right; left; exact h
      · right; right; left; exact h
      · right; right; right; exact h
    · exfalso
      have hlb := stage_lb hj
      have h5 : 5 * Q k ≤ Q j := by
        have := Q_mono (show k + 1 ≤ j by omega); rw [Q_succ] at this; exact this
      omega

lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hJ : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hJ
  obtain ⟨hn1, hn2⟩ := hJ
  have hp : 1 ≤ Q k := Q_one_le k
  have ha2 : 2 ≤ a := two_le_of_mem ha
  have hb2 : 2 ≤ b := two_le_of_mem hb
  have ha_lt : a < 10 * Q k := by omega
  have hb_lt : b < 10 * Q k := by omega
  rcases classify ha ha_lt with Ca | Ca | Ca | Ca
  · rcases classify hb hb_lt with Cb | Cb | Cb | Cb
    · exfalso; omega
    · exfalso; omega
    · exfalso; omega
    · exfalso; omega
  · rcases classify hb hb_lt with Cb | Cb | Cb | Cb
    · exfalso; omega
    · exfalso; omega
    · exact Or.inl ⟨Ca, mem_Icc.mpr ⟨Cb.1, Cb.2⟩⟩
    · exfalso; omega
  · rcases classify hb hb_lt with Cb | Cb | Cb | Cb
    · exfalso; omega
    · exact Or.inr ⟨Cb, mem_Icc.mpr ⟨Ca.1, Ca.2⟩⟩
    · exfalso; omega
    · exfalso; omega
  · exfalso; omega

lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hJ, hsum⟩
  rw [Set.mem_add] at hsum
  obtain ⟨a, ha, b, hb, hab⟩ := hsum
  have hJ' : a + b ∈ Jk k := by rw [hab]; exact hJ
  rcases rigidity (hT ha) (hT hb) hJ' with ⟨hac, _⟩ | ⟨hbc, _⟩
  · rw [hac] at ha; exact hck ha
  · rw [hbc] at hb; exact hck hb

lemma contra_side {k C : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T)
    (hCk : C < Q k) (hsyn : ∀ x : ℕ, ∃ m ∈ T + T, m ∈ Icc x (x + C)) : False := by
  have hgap := gap_lem hT hck
  obtain ⟨m, hmT, hmI⟩ := hsyn (9 * Q k)
  rw [mem_Icc] at hmI
  have hmJ : m ∈ Jk k := by
    rw [Jk, mem_Ico]; exact ⟨by omega, by omega⟩
  have hmem : m ∈ Jk k ∩ (T + T) := ⟨hmJ, hmT⟩
  rw [hgap] at hmem
  exact hmem

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn4
    have h := le_Q n
    exact basis_lem n n hn4 (by omega)
  · intro A₁ A₂ hA1 hA2 hcov hdisj hsyn
    obtain ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩ := hsyn
    set k := C₁ + C₂ + 1 with hk
    have hQk1 : C₁ < Q k := by have := le_Q k; omega
    have hQk2 : C₂ < Q k := by have := le_Q k; omega
    have hckA : ck k ∈ setA := ck_mem_setA k
    rcases hcov (ck k) hckA with h1 | h2
    · have hck2 : ck k ∉ A₂ := fun hc => by
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h1, hc⟩
        rw [hdisj] at hmem; exact hmem
      exact contra_side hA2 hck2 hQk2 hC2
    · have hck1 : ck k ∉ A₁ := fun hc => by
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hc, h2⟩
        rw [hdisj] at hmem; exact hmem
      exact contra_side hA1 hck1 hQk1 hC1

end Erdos741OAI

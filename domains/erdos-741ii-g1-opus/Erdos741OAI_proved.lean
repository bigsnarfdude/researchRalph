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

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q; exact pow_pos (by norm_num) k

lemma Q_one_le (k : ℕ) : 1 ≤ Q k := Q_pos k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp only [Q, pow_succ]; ring

lemma le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => exact Nat.zero_le _
  | succ n ih =>
    have hs := Q_succ n
    have hp := Q_pos n
    omega

lemma five_Q_le {i j : ℕ} (h : i < j) : 5 * Q i ≤ Q j := by
  have hp : Q (i + 1) ≤ Q j := by
    unfold Q
    exact Nat.pow_le_pow_right (by norm_num) h
  rw [Q_succ] at hp
  exact hp

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stageK (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stageK k

/-! ## Membership helpers -/

lemma mem_setA_of_stage {k x : ℕ} (h : x ∈ stageK k) : x ∈ setA := by
  simp only [setA, Set.mem_union, mem_iUnion]
  exact Or.inr ⟨k, h⟩

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by
  refine mem_setA_of_stage (k := k) ?_
  left; left
  exact Set.mem_singleton_iff.mpr rfl

lemma I_mem_setA {k y : ℕ} (h1 : 2 * Q k ≤ y) (h2 : y ≤ 3 * Q k) : y ∈ setA := by
  cases k with
  | zero =>
    have hq0 : Q 0 = 1 := by simp [Q]
    simp only [setA, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff, mem_iUnion]
    left; omega
  | succ k =>
    rw [Q_succ] at h1 h2
    refine mem_setA_of_stage (k := k) ?_
    simp only [stageK, Set.mem_union, Fk, mem_Icc]
    right
    constructor <;> omega

lemma B_mem_setA {k y : ℕ} (h1 : 5 * Q k ≤ y) (h2 : y ≤ 6 * Q k - 1) : y ∈ setA := by
  refine mem_setA_of_stage (k := k) ?_
  simp only [stageK, Set.mem_union, Bk, mem_Icc]
  left; right; exact ⟨h1, h2⟩

lemma F_mem_setA {k y : ℕ} (h1 : 10 * Q k - 1 ≤ y) (h2 : y ≤ 15 * Q k) : y ∈ setA := by
  refine mem_setA_of_stage (k := k) ?_
  simp only [stageK, Set.mem_union, Fk, mem_Icc]
  right; exact ⟨h1, h2⟩

lemma mem_setA_iff {x : ℕ} : x ∈ setA ↔ ((x = 2 ∨ x = 3) ∨ ∃ k, x ∈ stageK k) := by
  simp only [setA, Set.mem_union, Set.mem_insert_iff, Set.mem_singleton_iff, mem_iUnion]

lemma stage_bounds {j x : ℕ} (h : x ∈ stageK j) : 4 * Q j ≤ x ∧ x ≤ 15 * Q j := by
  have hq := Q_pos j
  simp only [stageK, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at h
  rcases h with (h | ⟨h1, h2⟩) | ⟨h1, h2⟩ <;> omega

lemma setA_cases {x : ℕ} (h : x ∈ setA) :
    (2 ≤ x ∧ x ≤ 3) ∨ ∃ j, 4 * Q j ≤ x ∧ x ≤ 15 * Q j ∧
      (x = 4 * Q j ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨ (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j)) := by
  rcases mem_setA_iff.mp h with (rfl | rfl) | ⟨j, hj⟩
  · left; omega
  · left; omega
  · right
    refine ⟨j, (stage_bounds hj).1, (stage_bounds hj).2, ?_⟩
    have hq := Q_pos j
    simp only [stageK, ck, Bk, Fk, Set.mem_union, Set.mem_singleton_iff, mem_Icc] at hj
    tauto

/-! ## Basis -/

lemma cover_q (q : ℕ) (hq : 1 ≤ q)
    (hI : ∀ y, 2 * q ≤ y → y ≤ 3 * q → y ∈ setA)
    (hck : 4 * q ∈ setA)
    (hB : ∀ y, 5 * q ≤ y → y ≤ 6 * q - 1 → y ∈ setA)
    (hF : ∀ y, 10 * q - 1 ≤ y → y ≤ 15 * q → y ∈ setA)
    {x : ℕ} (hx : 4 * q ≤ x) (hx2 : x ≤ 30 * q) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = x := by
  by_cases h6 : x ≤ 6 * q
  · by_cases h5 : x ≤ 5 * q
    · exact ⟨2 * q, hI _ (by omega) (by omega), x - 2 * q, hI _ (by omega) (by omega), by omega⟩
    · exact ⟨x - 3 * q, hI _ (by omega) (by omega), 3 * q, hI _ (by omega) (by omega), by omega⟩
  · by_cases h7 : x ≤ 7 * q
    · exact ⟨4 * q, hck, x - 4 * q, hI _ (by omega) (by omega), by omega⟩
    · by_cases h9m : x ≤ 9 * q - 1
      · by_cases h8 : x ≤ 8 * q - 1
        · exact ⟨x - 2 * q, hB _ (by omega) (by omega), 2 * q, hI _ (by omega) (by omega), by omega⟩
        · exact ⟨x - 3 * q, hB _ (by omega) (by omega), 3 * q, hI _ (by omega) (by omega), by omega⟩
      · by_cases h10m : x ≤ 10 * q - 1
        · exact ⟨4 * q, hck, x - 4 * q, hB _ (by omega) (by omega), by omega⟩
        · by_cases h12 : x ≤ 12 * q - 2
          · by_cases h11 : x ≤ 11 * q - 1
            · exact ⟨5 * q, hB _ (by omega) (by omega), x - 5 * q, hB _ (by omega) (by omega), by omega⟩
            · exact ⟨6 * q - 1, hB _ (by omega) (by omega), x - (6 * q - 1), hB _ (by omega) (by omega), by omega⟩
          · by_cases h18 : x ≤ 18 * q
            · by_cases h17 : x ≤ 17 * q
              · exact ⟨x - 2 * q, hF _ (by omega) (by omega), 2 * q, hI _ (by omega) (by omega), by omega⟩
              · exact ⟨x - 3 * q, hF _ (by omega) (by omega), 3 * q, hI _ (by omega) (by omega), by omega⟩
            · by_cases h21 : x ≤ 21 * q - 1
              · by_cases h20 : x ≤ 20 * q
                · exact ⟨x - 5 * q, hF _ (by omega) (by omega), 5 * q, hB _ (by omega) (by omega), by omega⟩
                · exact ⟨x - (6 * q - 1), hF _ (by omega) (by omega), 6 * q - 1, hB _ (by omega) (by omega), by omega⟩
              · by_cases h25 : x ≤ 25 * q - 1
                · exact ⟨10 * q - 1, hF _ (by omega) (by omega), x - (10 * q - 1), hF _ (by omega) (by omega), by omega⟩
                · exact ⟨15 * q, hF _ (by omega) (by omega), x - 15 * q, hF _ (by omega) (by omega), by omega⟩

lemma cover_level (k : ℕ) {x : ℕ} (hx : 4 * Q k ≤ x) (hx2 : x ≤ 30 * Q k) :
    ∃ a ∈ setA, ∃ b ∈ setA, a + b = x :=
  cover_q (Q k) (Q_one_le k)
    (fun y h1 h2 => I_mem_setA h1 h2)
    (show 4 * Q k ∈ setA from ck_mem_setA k)
    (fun y h1 h2 => B_mem_setA h1 h2)
    (fun y h1 h2 => F_mem_setA h1 h2)
    hx hx2

lemma basis_range (k : ℕ) :
    ∀ {x : ℕ}, 4 ≤ x → x ≤ 30 * Q k → ∃ a ∈ setA, ∃ b ∈ setA, a + b = x := by
  induction k with
  | zero =>
    intro x hx hx2
    have hb1 : 4 * Q 0 ≤ x := by simp only [Q, pow_zero, mul_one]; omega
    exact cover_level 0 hb1 hx2
  | succ k ih =>
    intro x hx hx2
    by_cases hxk : x ≤ 30 * Q k
    · exact ih hx hxk
    · push_neg at hxk
      have hb1 : 4 * Q (k + 1) ≤ x := by rw [Q_succ]; omega
      exact cover_level (k + 1) hb1 hx2

lemma basis_all {n : ℕ} (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have h1 := le_Q n
  have h2 := Q_pos n
  exact basis_range n hn (by omega)

/-! ## Rigidity and gap -/

lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hlo : 9 * Q k ≤ a + b) (hhi : a + b < 10 * Q k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hqk := Q_pos k
  rcases setA_cases ha with ⟨ha2, ha3⟩ | ⟨i, hai_lo, hai_hi, hai3⟩
  · -- a tiny
    exfalso
    rcases setA_cases hb with ⟨hb2, hb3⟩ | ⟨j, hbj_lo, hbj_hi, hbj3⟩
    · omega
    · have hqj := Q_pos j
      rcases lt_trichotomy j k with hj | hj | hj
      · have h5 := five_Q_le hj; omega
      · rw [hj] at hbj_lo hbj_hi hbj3
        rcases hbj3 with h | ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> omega
      · have h5 := five_Q_le hj; omega
  · -- a in stage i
    have hqi := Q_pos i
    rcases setA_cases hb with ⟨hb2, hb3⟩ | ⟨j, hbj_lo, hbj_hi, hbj3⟩
    · -- b tiny
      exfalso
      rcases lt_trichotomy i k with hi | hi | hi
      · have h5 := five_Q_le hi; omega
      · rw [hi] at hai_lo hai_hi hai3
        rcases hai3 with h | ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> omega
      · have h5 := five_Q_le hi; omega
    · -- both stage
      have hqj := Q_pos j
      have ha_lt : a < 10 * Q k := by omega
      have hb_lt : b < 10 * Q k := by omega
      rcases lt_trichotomy i k with hi | hi | hi
      · -- i < k
        exfalso
        have h5i := five_Q_le hi
        rcases lt_trichotomy j k with hj | hj | hj
        · have h5j := five_Q_le hj; omega
        · rw [hj] at hbj_lo hbj_hi hbj3
          rcases hbj3 with h | ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> omega
        · have h5j := five_Q_le hj; omega
      · -- i = k
        rw [hi] at hai_lo hai_hi hai3
        rcases lt_trichotomy j k with hj | hj | hj
        · -- j < k
          exfalso
          have h5j := five_Q_le hj
          rcases hai3 with h | ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> omega
        · -- j = k : main case
          rw [hj] at hbj_lo hbj_hi hbj3
          rcases hai3 with ha4 | ⟨haB1, haB2⟩ | ⟨haF1, haF2⟩
          · -- a = 4 Q k
            left
            refine ⟨by simp only [ck]; omega, ?_⟩
            simp only [Bk, mem_Icc]
            rcases hbj3 with hb4 | ⟨hbB1, hbB2⟩ | ⟨hbF1, hbF2⟩ <;> omega
          · -- a in Bk
            rcases hbj3 with hb4 | ⟨hbB1, hbB2⟩ | ⟨hbF1, hbF2⟩
            · right
              refine ⟨by simp only [ck]; omega, ?_⟩
              simp only [Bk, mem_Icc]; omega
            · exfalso; omega
            · exfalso; omega
          · -- a in Fk
            exfalso; omega
        · -- j > k
          exfalso
          have h5j := five_Q_le hj; omega
      · -- i > k
        exfalso
        have h5i := five_Q_le hi; omega

lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T) :
    ∀ n, n ∈ Jk k → n ∉ (T + T) := by
  intro n hn hmem
  rw [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn1, hn2⟩ := hn
  have hrig := rigidity (k := k) (hT ha) (hT hb) (by omega) (by omega)
  rcases hrig with ⟨ha_ck, _⟩ | ⟨hb_ck, _⟩
  · exact hck (ha_ck ▸ ha)
  · exact hck (hb_ck ▸ hb)

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
  · intro n hn
    exact basis_all hn
  · intro A₁ A₂ hA1 hA2 hcover hdisj hsyn
    obtain ⟨h1s, h2s⟩ := hsyn
    obtain ⟨C1, hC1⟩ := h1s
    obtain ⟨C2, hC2⟩ := h2s
    obtain ⟨k, hk1, hk2⟩ : ∃ k, C1 < Q k ∧ C2 < Q k := by
      refine ⟨C1 + C2 + 1, ?_, ?_⟩
      · have h := le_Q (C1 + C2 + 1); omega
      · have h := le_Q (C1 + C2 + 1); omega
    have hckA : ck k ∈ setA := ck_mem_setA k
    rcases hcover (ck k) hckA with h1 | h2
    · have hnotin : ck k ∉ A₂ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h1, hc⟩
        rw [hdisj] at hmem
        simp at hmem
      obtain ⟨m, hmAdd, hmIcc⟩ := hC2 (9 * Q k)
      simp only [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      exact gap_lem hA2 hnotin m hmJ hmAdd
    · have hnotin : ck k ∉ A₁ := by
        intro hc
        have hmem : ck k ∈ A₁ ∩ A₂ := ⟨hc, h2⟩
        rw [hdisj] at hmem
        simp at hmem
      obtain ⟨m, hmAdd, hmIcc⟩ := hC1 (9 * Q k)
      simp only [mem_Icc] at hmIcc
      have hmJ : m ∈ Jk k := by
        simp only [Jk, mem_Ico]; omega
      exact gap_lem hA1 hnotin m hmJ hmAdd

end Erdos741OAI

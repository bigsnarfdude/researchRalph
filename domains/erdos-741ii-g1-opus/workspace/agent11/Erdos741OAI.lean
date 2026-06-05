import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def stage (k : ℕ) : Set ℕ := {ck k} ∪ Bk k ∪ Fk k
def setA : Set ℕ := {2, 3} ∪ ⋃ k, stage k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k + 1) => Akn k ∪ stage k

-- Q basic facts
lemma Q_pos (k : ℕ) : 0 < Q k := pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma Q_mono {j k : ℕ} (h : j ≤ k) : Q j ≤ Q k :=
  Nat.pow_le_pow_right (by norm_num) h

-- sum membership helper
lemma mk_sum {x a b : ℕ} {S T : Set ℕ} (ha : a ∈ S) (hb : b ∈ T) (h : a + b = x) :
    x ∈ S + T :=
  Set.mem_add.mpr ⟨a, ha, b, hb, h⟩

-- stage / Akn membership
lemma ck_mem_stage (k : ℕ) : ck k ∈ stage k := Or.inl (Or.inl rfl)

lemma Bk_mem_stage {k x : ℕ} (h : x ∈ Bk k) : x ∈ stage k := Or.inl (Or.inr h)

lemma Fk_mem_stage {k x : ℕ} (h : x ∈ Fk k) : x ∈ stage k := Or.inr h

lemma stage_subset_Akn {k : ℕ} : stage k ⊆ Akn (k + 1) := fun _ hx => Or.inr hx

lemma Akn_mono {k : ℕ} : Akn k ⊆ Akn (k + 1) := fun _ hx => Or.inl hx

lemma Akn_le {j k : ℕ} (h : j ≤ k) : Akn j ⊆ Akn k := by
  induction h with
  | refl => exact subset_rfl
  | step _ ih => exact ih.trans Akn_mono

lemma Akn_subset_setA {k : ℕ} : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Or.inr (mem_iUnion.mpr ⟨k, hx⟩)

-- Membership helpers (interval bounds in terms of Q k)
lemma mem_Ik {k y : ℕ} (h1 : 10 * Q k - 1 ≤ y) (h2 : y ≤ 15 * Q k) : y ∈ Fk k :=
  mem_Icc.mpr ⟨h1, h2⟩

lemma mem_Bkp {k y : ℕ} (h1 : 25 * Q k ≤ y) (h2 : y ≤ 30 * Q k - 1) : y ∈ Bk (k + 1) := by
  show y ∈ Icc (5 * Q (k + 1)) (6 * Q (k + 1) - 1)
  rw [mem_Icc, Q_succ]; omega

lemma mem_Fkp {k y : ℕ} (h1 : 50 * Q k - 1 ≤ y) (h2 : y ≤ 75 * Q k) : y ∈ Fk (k + 1) := by
  show y ∈ Icc (10 * Q (k + 1) - 1) (15 * Q (k + 1))
  rw [mem_Icc, Q_succ]; omega

lemma ckp_val (k : ℕ) : ck (k + 1) = 20 * Q k := by
  show 4 * Q (k + 1) = 20 * Q k
  rw [Q_succ]; ring

-- Lift interval memberships into Akn (k+2)
lemma Ik_in (k : ℕ) {y : ℕ} (h : y ∈ Fk k) : y ∈ Akn (k + 2) :=
  Akn_le (by omega) (stage_subset_Akn (Fk_mem_stage h))

lemma ckp_in (k : ℕ) : ck (k + 1) ∈ Akn (k + 2) :=
  stage_subset_Akn (ck_mem_stage (k + 1))

lemma Bkp_in (k : ℕ) {y : ℕ} (h : y ∈ Bk (k + 1)) : y ∈ Akn (k + 2) :=
  stage_subset_Akn (Bk_mem_stage h)

lemma Fkp_in (k : ℕ) {y : ℕ} (h : y ∈ Fk (k + 1)) : y ∈ Akn (k + 2) :=
  stage_subset_Akn (Fk_mem_stage h)

-- The eight-pair covering: covers [4 Q(k+1), 30 Q(k+1)] using stage (k+1) and Fk k.
lemma cover_top (k : ℕ) :
    Icc (4 * Q (k + 1)) (30 * Q (k + 1)) ⊆ Akn (k + 2) + Akn (k + 2) := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  rw [Q_succ] at hlo hhi
  have hq := Q_pos k
  have hckp : ck (k + 1) = 20 * Q k := ckp_val k
  -- cascade of segments
  by_cases hc1 : x ≤ 25 * Q k
  · exact Set.mem_add.mpr ⟨10 * Q k, Ik_in k (mem_Ik (by omega) (by omega)),
      x - 10 * Q k, Ik_in k (mem_Ik (by omega) (by omega)), by omega⟩
  by_cases hc2 : x ≤ 30 * Q k
  · exact Set.mem_add.mpr ⟨15 * Q k, Ik_in k (mem_Ik (by omega) (by omega)),
      x - 15 * Q k, Ik_in k (mem_Ik (by omega) (by omega)), by omega⟩
  by_cases hc3 : x ≤ 35 * Q k
  · exact Set.mem_add.mpr ⟨ck (k + 1), ckp_in k,
      x - ck (k + 1), Ik_in k (mem_Ik (by omega) (by omega)), by omega⟩
  by_cases hc4 : x ≤ 40 * Q k - 1
  · exact Set.mem_add.mpr ⟨10 * Q k, Ik_in k (mem_Ik (by omega) (by omega)),
      x - 10 * Q k, Bkp_in k (mem_Bkp (by omega) (by omega)), by omega⟩
  by_cases hc5 : x ≤ 45 * Q k - 1
  · exact Set.mem_add.mpr ⟨15 * Q k, Ik_in k (mem_Ik (by omega) (by omega)),
      x - 15 * Q k, Bkp_in k (mem_Bkp (by omega) (by omega)), by omega⟩
  by_cases hc6 : x ≤ 50 * Q k - 1
  · exact Set.mem_add.mpr ⟨ck (k + 1), ckp_in k,
      x - ck (k + 1), Bkp_in k (mem_Bkp (by omega) (by omega)), by omega⟩
  by_cases hc7 : x ≤ 55 * Q k - 1
  · exact Set.mem_add.mpr ⟨25 * Q k, Bkp_in k (mem_Bkp (by omega) (by omega)),
      x - 25 * Q k, Bkp_in k (mem_Bkp (by omega) (by omega)), by omega⟩
  by_cases hc8 : x ≤ 60 * Q k - 2
  · exact Set.mem_add.mpr ⟨30 * Q k - 1, Bkp_in k (mem_Bkp (by omega) (by omega)),
      x - (30 * Q k - 1), Bkp_in k (mem_Bkp (by omega) (by omega)), by omega⟩
  by_cases hc9 : x ≤ 85 * Q k - 1
  · exact Set.mem_add.mpr ⟨10 * Q k - 1, Ik_in k (mem_Ik (by omega) (by omega)),
      x - (10 * Q k - 1), Fkp_in k (mem_Fkp (by omega) (by omega)), by omega⟩
  by_cases hc10 : x ≤ 90 * Q k
  · exact Set.mem_add.mpr ⟨15 * Q k, Ik_in k (mem_Ik (by omega) (by omega)),
      x - 15 * Q k, Fkp_in k (mem_Fkp (by omega) (by omega)), by omega⟩
  by_cases hc11 : x ≤ 105 * Q k - 1
  · exact Set.mem_add.mpr ⟨30 * Q k - 1, Bkp_in k (mem_Bkp (by omega) (by omega)),
      x - (30 * Q k - 1), Fkp_in k (mem_Fkp (by omega) (by omega)), by omega⟩
  by_cases hc12 : x ≤ 125 * Q k - 1
  · exact Set.mem_add.mpr ⟨50 * Q k - 1, Fkp_in k (mem_Fkp (by omega) (by omega)),
      x - (50 * Q k - 1), Fkp_in k (mem_Fkp (by omega) (by omega)), by omega⟩
  · exact Set.mem_add.mpr ⟨75 * Q k, Fkp_in k (mem_Fkp (by omega) (by omega)),
      x - 75 * Q k, Fkp_in k (mem_Fkp (by omega) (by omega)), by omega⟩

-- Base case (level 0) membership helpers: elements of Akn 1
lemma two_mem_Akn1 : (2 : ℕ) ∈ Akn 1 := Or.inl (by show (2:ℕ) ∈ ({2,3}:Set ℕ); simp)
lemma three_mem_Akn1 : (3 : ℕ) ∈ Akn 1 := Or.inl (by show (3:ℕ) ∈ ({2,3}:Set ℕ); simp)

lemma four_mem_Akn1 : (4 : ℕ) ∈ Akn 1 := by
  have h := stage_subset_Akn (ck_mem_stage 0)
  simp only [ck, Q, pow_zero, mul_one] at h
  exact h

lemma five_mem_Akn1 : (5 : ℕ) ∈ Akn 1 := by
  have h : (5 : ℕ) ∈ Bk 0 := by
    show (5:ℕ) ∈ Icc (5 * Q 0) (6 * Q 0 - 1); simp only [Q, pow_zero, mul_one, mem_Icc]; omega
  exact stage_subset_Akn (Bk_mem_stage h)

lemma f0_mem_Akn1 {y : ℕ} (h1 : 9 ≤ y) (h2 : y ≤ 15) : y ∈ Akn 1 := by
  have h : y ∈ Fk 0 := by
    show y ∈ Icc (10 * Q 0 - 1) (15 * Q 0); simp only [Q, pow_zero, mul_one, mem_Icc]; omega
  exact stage_subset_Akn (Fk_mem_stage h)

lemma n_le_Q (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => exact Nat.zero_le _
  | succ k ih =>
    have hq := Q_pos k
    rw [Q_succ]; omega

-- Basis covering by induction: Icc 4 (6 Q(k+1)) ⊆ Akn(k+1) + Akn(k+1)
lemma basis_cover (k : ℕ) : Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    rw [mem_Icc] at hx
    obtain ⟨hlo, hhi⟩ := hx
    simp only [Q, pow_one] at hhi   -- hhi : x ≤ 6 * 5 = 30
    by_cases h5 : x ≤ 5
    · exact Set.mem_add.mpr ⟨2, two_mem_Akn1, x - 2,
        (by rcases (by omega : x - 2 = 2 ∨ x - 2 = 3) with h | h <;> rw [h]
            <;> [exact two_mem_Akn1; exact three_mem_Akn1]), by omega⟩
    by_cases h6 : x ≤ 6
    · exact Set.mem_add.mpr ⟨3, three_mem_Akn1, 3, three_mem_Akn1, by omega⟩
    by_cases h7 : x ≤ 7
    · exact Set.mem_add.mpr ⟨3, three_mem_Akn1, 4, four_mem_Akn1, by omega⟩
    by_cases h8 : x ≤ 8
    · exact Set.mem_add.mpr ⟨4, four_mem_Akn1, 4, four_mem_Akn1, by omega⟩
    by_cases h9 : x ≤ 9
    · exact Set.mem_add.mpr ⟨4, four_mem_Akn1, 5, five_mem_Akn1, by omega⟩
    by_cases h10 : x ≤ 10
    · exact Set.mem_add.mpr ⟨5, five_mem_Akn1, 5, five_mem_Akn1, by omega⟩
    by_cases h17 : x ≤ 17
    · exact Set.mem_add.mpr ⟨2, two_mem_Akn1, x - 2, f0_mem_Akn1 (by omega) (by omega), by omega⟩
    by_cases h24 : x ≤ 24
    · exact Set.mem_add.mpr ⟨9, f0_mem_Akn1 (by omega) (by omega),
        x - 9, f0_mem_Akn1 (by omega) (by omega), by omega⟩
    · exact Set.mem_add.mpr ⟨15, f0_mem_Akn1 (by omega) (by omega),
        x - 15, f0_mem_Akn1 (by omega) (by omega), by omega⟩
  | succ k ih =>
    intro x hx
    rw [mem_Icc] at hx
    obtain ⟨hlo, hhi⟩ := hx
    by_cases hsplit : x ≤ 6 * Q (k + 1)
    · have hmem : x ∈ Akn (k + 1) + Akn (k + 1) := ih (mem_Icc.mpr ⟨hlo, hsplit⟩)
      rcases Set.mem_add.mp hmem with ⟨a, ha, b, hb, hab⟩
      exact Set.mem_add.mpr ⟨a, Akn_mono ha, b, Akn_mono hb, hab⟩
    · push_neg at hsplit
      have hx2 : x ∈ Icc (4 * Q (k + 1)) (30 * Q (k + 1)) := by
        rw [mem_Icc]
        have hp := Q_pos (k + 1)
        have heq : 6 * Q (k + 1 + 1) = 30 * Q (k + 1) := by rw [Q_succ]; ring
        constructor
        · omega
        · omega
      exact cover_top k hx2

-- Cross-stage Q comparisons
lemma Q_lt {j k : ℕ} (h : j < k) : 5 * Q j ≤ Q k := by
  have hh : Q (j + 1) ≤ Q k := Q_mono h
  rw [Q_succ] at hh; exact hh

lemma Q_gt {j k : ℕ} (h : k < j) : 5 * Q k ≤ Q j := by
  have hh : Q (k + 1) ≤ Q j := Q_mono h
  rw [Q_succ] at hh; exact hh

-- Classify a member of setA
lemma setA_cases {a : ℕ} (ha : a ∈ setA) :
    (a = 2 ∨ a = 3) ∨ ∃ j, a = ck j ∨ a ∈ Bk j ∨ a ∈ Fk j := by
  rcases ha with h | h
  · left; simpa using h
  · right
    rw [mem_iUnion] at h
    obtain ⟨j, hj⟩ := h
    refine ⟨j, ?_⟩
    rcases hj with (hc | hb) | hf
    · exact Or.inl (mem_singleton_iff.mp hc)
    · exact Or.inr (Or.inl hb)
    · exact Or.inr (Or.inr hf)

lemma setA_band {x : ℕ} (hx : x ∈ setA) :
    x ≤ 3 ∨ ∃ j, (x = 4 * Q j) ∨ (5 * Q j ≤ x ∧ x ≤ 6 * Q j - 1) ∨
      (10 * Q j - 1 ≤ x ∧ x ≤ 15 * Q j) := by
  rcases setA_cases hx with (h | h) | ⟨j, h⟩
  · left; omega
  · left; omega
  · right; refine ⟨j, ?_⟩
    rcases h with h | h | h
    · left; exact h
    · right; left; simpa [Bk, mem_Icc] using h
    · right; right; simpa [Fk, mem_Icc] using h

lemma setA_ge_two {a : ℕ} (ha : a ∈ setA) : 2 ≤ a := by
  rcases setA_band ha with h | ⟨j, h⟩
  · -- could be small; but actually elements are ≥2; for ≤3 case need a≥2.
    -- setA_band's small branch only guarantees ≤3, so handle via setA_cases instead.
    rcases setA_cases ha with (h2 | h2) | ⟨i, h2⟩
    · omega
    · omega
    · have hq := Q_pos i
      rcases h2 with h2 | h2 | h2
      · show 2 ≤ a; rw [h2]; show 2 ≤ 4 * Q i; omega
      · simp only [Bk, mem_Icc] at h2; omega
      · simp only [Fk, mem_Icc] at h2; omega
  · have hq := Q_pos j
    rcases h with h | h | h
    · omega
    · omega
    · omega

-- For x ∈ setA with x < 10 Q k, reduce to four possibilities
lemma elt_lt10 {k x : ℕ} (hx : x ∈ setA) (hlt : x < 10 * Q k) :
    x ≤ 3 * Q k ∨ x = 4 * Q k ∨ (5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1) ∨ x = 10 * Q k - 1 := by
  have hq := Q_pos k
  rcases setA_band hx with h | ⟨j, h⟩
  · left; omega
  · rcases lt_trichotomy j k with hjk | hjk | hjk
    · have h5 : 5 * Q j ≤ Q k := Q_lt hjk
      left
      rcases h with h | h | h
      · omega
      · omega
      · omega
    · rw [hjk] at h
      rcases h with h | h | h
      · right; left; exact h
      · right; right; left; exact h
      · right; right; right; omega
    · exfalso
      have h5 : 5 * Q k ≤ Q j := Q_gt hjk
      rcases h with h | h | h <;> omega

-- Rigidity: any decomposition of n ∈ Jk k is ck k + (something in Bk k)
lemma rigidity {k a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA)
    (hn : a + b ∈ Jk k) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  have hq := Q_pos k
  rw [Jk, mem_Ico] at hn
  obtain ⟨hsl, hsh⟩ := hn
  have ha2 := setA_ge_two ha
  have hb2 := setA_ge_two hb
  have ha10 : a < 10 * Q k := by omega
  have hb10 : b < 10 * Q k := by omega
  rcases elt_lt10 ha ha10 with hA | hA | hA | hA <;>
    rcases elt_lt10 hb hb10 with hB | hB | hB | hB
  -- a ≤ 3Qk
  · exfalso; omega
  · exfalso; omega
  · exfalso; omega
  · exfalso; omega
  -- a = 4Qk
  · exfalso; omega
  · exfalso; omega
  · exact Or.inl ⟨hA, mem_Icc.mpr ⟨hB.1, hB.2⟩⟩
  · exfalso; omega
  -- a ∈ Bk
  · exfalso; omega
  · exact Or.inr ⟨hB, mem_Icc.mpr ⟨hA.1, hA.2⟩⟩
  · exfalso; omega
  · exfalso; omega
  -- a = 10Qk-1
  · exfalso; omega
  · exfalso; omega
  · exfalso; omega
  · exfalso; omega

-- Gap lemma: if ck k ∉ T, then T+T misses Jk k
lemma gap_lem {k : ℕ} {T : Set ℕ} (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, mem_empty_iff_false, iff_false]
  rintro ⟨hjn, hsum⟩
  rcases Set.mem_add.mp hsum with ⟨a, ha, b, hb, hab⟩
  have hn : a + b ∈ Jk k := by rw [hab]; exact hjn
  rcases rigidity (hT ha) (hT hb) hn with ⟨hac, _⟩ | ⟨hbc, _⟩
  · exact hck (hac ▸ ha)
  · exact hck (hbc ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · -- basis property
    intro n hn
    have hmem : n ∈ Icc 4 (6 * Q (n + 1)) := by
      rw [mem_Icc]
      refine ⟨hn, ?_⟩
      have h1 : n ≤ Q n := n_le_Q n
      have h2 : Q n ≤ Q (n + 1) := Q_mono (by omega)
      have h3 := Q_pos (n + 1)
      omega
    have hcov := basis_cover n hmem
    rcases Set.mem_add.mp hcov with ⟨a, ha, b, hb, hab⟩
    exact ⟨a, Akn_subset_setA ha, b, Akn_subset_setA hb, hab⟩
  · -- no partition is both-syndetic
    intro A₁ A₂ h1 h2 hcov hdisj
    rintro ⟨⟨C₁, hC1⟩, ⟨C₂, hC2⟩⟩
    set k := C₁ + C₂ + 1 with hk
    have hkk : k ≤ Q k := n_le_Q k
    have hckA : ck k ∈ setA := Or.inr (mem_iUnion.mpr ⟨k, ck_mem_stage k⟩)
    rcases hcov (ck k) hckA with hin | hin
    · have hnotin : ck k ∉ A₂ := by
        intro hcon
        have hmm : ck k ∈ A₁ ∩ A₂ := ⟨hin, hcon⟩
        rw [hdisj] at hmm; exact hmm
      have hgap := gap_lem h2 hnotin
      obtain ⟨m, hmS, hmI⟩ := hC2 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]
        refine ⟨by omega, ?_⟩
        have hlt : C₂ < Q k := by omega
        omega
      have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra; exact hcontra
    · have hnotin : ck k ∉ A₁ := by
        intro hcon
        have hmm : ck k ∈ A₁ ∩ A₂ := ⟨hcon, hin⟩
        rw [hdisj] at hmm; exact hmm
      have hgap := gap_lem h1 hnotin
      obtain ⟨m, hmS, hmI⟩ := hC1 (9 * Q k)
      rw [mem_Icc] at hmI
      have hmJ : m ∈ Jk k := by
        rw [Jk, mem_Ico]
        refine ⟨by omega, ?_⟩
        have hlt : C₁ < Q k := by omega
        omega
      have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmS⟩
      rw [hgap] at hcontra; exact hcontra

end Erdos741OAI

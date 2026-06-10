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

-- The construction
def Q (k : ℕ) : ℕ := 5 ^ k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | (k+1) => Akn k ∪ ({ck k} ∪ Bk k ∪ Fk k)

-- Basic facts about Q
lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k

lemma Q_succ (k : ℕ) : Q (k+1) = 5 * Q k := by unfold Q; rw [pow_succ]; ring

lemma lt_Q (k : ℕ) : k < Q k := by
  induction k with
  | zero => unfold Q; norm_num
  | succ n ih =>
    unfold Q at ih ⊢
    have hp : 1 ≤ 5 ^ n := Nat.one_le_pow n 5 (by norm_num)
    rw [pow_succ]
    omega

-- Monotonicity & containment
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k+1) := by
  intro x hx
  exact Or.inl hx

lemma akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    rcases hx with hx | hx
    · exact ih hx
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, hx⟩)

lemma ck_mem (k : ℕ) : ck k ∈ setA :=
  Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)

-- Subset lemmas into Akn (k+2)
lemma Fkk_sub (k : ℕ) : Fk k ⊆ Akn (k+2) := by
  intro x hx
  apply akn_mono (k+1)
  exact Or.inr (Or.inr hx)

lemma ck1_mem (k : ℕ) : ck (k+1) ∈ Akn (k+2) :=
  Or.inr (Or.inl (Or.inl rfl))

lemma Bk1_sub (k : ℕ) : Bk (k+1) ⊆ Akn (k+2) :=
  fun x hx => Or.inr (Or.inl (Or.inr hx))

lemma Fk1_sub (k : ℕ) : Fk (k+1) ⊆ Akn (k+2) :=
  fun x hx => Or.inr (Or.inr hx)

-- Membership helpers (level k+1 scale)
lemma mem_Fkk {k a : ℕ} (h1 : 10 * Q k - 1 ≤ a) (h2 : a ≤ 15 * Q k) : a ∈ Akn (k+2) :=
  Fkk_sub k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_ck1 {k a : ℕ} (h : a = 4 * Q (k+1)) : a ∈ Akn (k+2) := by
  rw [h]; exact ck1_mem k

lemma mem_Bk1 {k a : ℕ} (h1 : 5 * Q (k+1) ≤ a) (h2 : a ≤ 6 * Q (k+1) - 1) : a ∈ Akn (k+2) :=
  Bk1_sub k (mem_Icc.mpr ⟨h1, h2⟩)

lemma mem_Fk1 {k a : ℕ} (h1 : 10 * Q (k+1) - 1 ≤ a) (h2 : a ≤ 15 * Q (k+1)) : a ∈ Akn (k+2) :=
  Fk1_sub k (mem_Icc.mpr ⟨h1, h2⟩)

-- Membership helpers for Akn 1 (base case)
lemma mem_I0 {a : ℕ} (h1 : 2 ≤ a) (h2 : a ≤ 3) : a ∈ Akn 1 := by
  apply akn_mono 0
  show a ∈ ({2, 3} : Set ℕ)
  rw [Set.mem_insert_iff, Set.mem_singleton_iff]
  omega

lemma mem_c0 : (4 : ℕ) ∈ Akn 1 := by
  have h4 : (4 : ℕ) = ck 0 := by norm_num [ck, Q]
  rw [h4]
  exact Or.inr (Or.inl (Or.inl rfl))

lemma mem_B0 {a : ℕ} (h : a = 5) : a ∈ Akn 1 := by
  rw [h]
  refine Or.inr (Or.inl (Or.inr ?_))
  simp only [Bk, mem_Icc]
  have : Q 0 = 1 := rfl
  rw [this]; omega

lemma mem_F0 {a : ℕ} (h1 : 9 ≤ a) (h2 : a ≤ 15) : a ∈ Akn 1 := by
  refine Or.inr (Or.inr ?_)
  simp only [Fk, mem_Icc]
  have : Q 0 = 1 := rfl
  rw [this]; omega

-- The eight-pair cover at level k+1
lemma eight_cover (k : ℕ) :
    Icc (4 * Q (k+1)) (30 * Q (k+1)) ⊆ Akn (k+2) + Akn (k+2) := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  have hs : Q (k+1) = 5 * Q k := Q_succ k
  have hpos : 0 < Q k := Q_pos k
  by_cases c1 : x ≤ 30 * Q k
  · -- I+I
    exact Set.mem_add.mpr ⟨x - max (10 * Q k - 1) (x - 15 * Q k), mem_Fkk (by omega) (by omega),
      max (10 * Q k - 1) (x - 15 * Q k), mem_Fkk (by omega) (by omega), by omega⟩
  by_cases c2 : x ≤ 35 * Q k
  · -- I+c
    exact Set.mem_add.mpr ⟨4 * Q (k+1), mem_ck1 rfl,
      x - 4 * Q (k+1), mem_Fkk (by omega) (by omega), by omega⟩
  by_cases c3 : x ≤ 45 * Q k - 1
  · -- I+B
    exact Set.mem_add.mpr ⟨x - max (5 * Q (k+1)) (x - 15 * Q k), mem_Fkk (by omega) (by omega),
      max (5 * Q (k+1)) (x - 15 * Q k), mem_Bk1 (by omega) (by omega), by omega⟩
  by_cases c4 : x ≤ 50 * Q k - 1
  · -- c+B
    exact Set.mem_add.mpr ⟨4 * Q (k+1), mem_ck1 rfl,
      x - 4 * Q (k+1), mem_Bk1 (by omega) (by omega), by omega⟩
  by_cases c5 : x ≤ 60 * Q k - 2
  · -- B+B
    exact Set.mem_add.mpr
      ⟨x - max (5 * Q (k+1)) (x - (6 * Q (k+1) - 1)), mem_Bk1 (by omega) (by omega),
       max (5 * Q (k+1)) (x - (6 * Q (k+1) - 1)), mem_Bk1 (by omega) (by omega), by omega⟩
  by_cases c6 : x ≤ 90 * Q k
  · -- I+F
    exact Set.mem_add.mpr
      ⟨x - max (10 * Q (k+1) - 1) (x - 15 * Q k), mem_Fkk (by omega) (by omega),
       max (10 * Q (k+1) - 1) (x - 15 * Q k), mem_Fk1 (by omega) (by omega), by omega⟩
  by_cases c7 : x ≤ 105 * Q k - 1
  · -- B+F
    exact Set.mem_add.mpr
      ⟨x - max (10 * Q (k+1) - 1) (x - (6 * Q (k+1) - 1)), mem_Bk1 (by omega) (by omega),
       max (10 * Q (k+1) - 1) (x - (6 * Q (k+1) - 1)), mem_Fk1 (by omega) (by omega), by omega⟩
  · -- F+F
    exact Set.mem_add.mpr
      ⟨x - max (10 * Q (k+1) - 1) (x - 15 * Q (k+1)), mem_Fk1 (by omega) (by omega),
       max (10 * Q (k+1) - 1) (x - 15 * Q (k+1)), mem_Fk1 (by omega) (by omega), by omega⟩

-- Base case cover: Icc 4 30 ⊆ Akn 1 + Akn 1
lemma base_cover : Icc 4 30 ⊆ Akn 1 + Akn 1 := by
  intro x hx
  rw [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  by_cases b1 : x ≤ 6
  · -- I'+I'
    exact Set.mem_add.mpr ⟨x - max 2 (x - 3), mem_I0 (by omega) (by omega),
      max 2 (x - 3), mem_I0 (by omega) (by omega), by omega⟩
  by_cases b2 : x ≤ 7
  · -- I'+c'
    exact Set.mem_add.mpr ⟨x - 4, mem_I0 (by omega) (by omega), 4, mem_c0, by omega⟩
  by_cases b3 : x ≤ 8
  · -- I'+B'
    exact Set.mem_add.mpr ⟨x - 5, mem_I0 (by omega) (by omega), 5, mem_B0 rfl, by omega⟩
  by_cases b4 : x ≤ 9
  · -- c'+B'
    exact Set.mem_add.mpr ⟨4, mem_c0, 5, mem_B0 rfl, by omega⟩
  by_cases b5 : x ≤ 10
  · -- B'+B'
    exact Set.mem_add.mpr ⟨5, mem_B0 rfl, 5, mem_B0 rfl, by omega⟩
  by_cases b6 : x ≤ 18
  · -- I'+F'
    exact Set.mem_add.mpr ⟨x - max 9 (x - 3), mem_I0 (by omega) (by omega),
      max 9 (x - 3), mem_F0 (by omega) (by omega), by omega⟩
  by_cases b7 : x ≤ 20
  · -- B'+F'
    exact Set.mem_add.mpr ⟨5, mem_B0 rfl, x - 5, mem_F0 (by omega) (by omega), by omega⟩
  · -- F'+F'
    exact Set.mem_add.mpr ⟨x - max 9 (x - 15), mem_F0 (by omega) (by omega),
      max 9 (x - 15), mem_F0 (by omega) (by omega), by omega⟩

-- Total cover by induction
lemma cover_total (k : ℕ) : Icc 4 (30 * Q k) ⊆ Akn (k+1) + Akn (k+1) := by
  induction k with
  | zero =>
    have h30 : 30 * Q 0 = 30 := by simp [Q]
    rw [h30]
    exact base_cover
  | succ k ih =>
    intro x hx
    rw [mem_Icc] at hx
    by_cases hsmall : x ≤ 30 * Q k
    · have hmono : Akn (k+1) + Akn (k+1) ⊆ Akn (k+2) + Akn (k+2) :=
        Set.add_subset_add (akn_mono (k+1)) (akn_mono (k+1))
      exact hmono (ih (mem_Icc.mpr ⟨hx.1, hsmall⟩))
    · push_neg at hsmall
      have hx' : x ∈ Icc (4 * Q (k+1)) (30 * Q (k+1)) := by
        rw [mem_Icc]
        refine ⟨?_, hx.2⟩
        have hs := Q_succ k; have := Q_pos k; omega
      exact eight_cover k hx'

-- The basis property
lemma basis (n : ℕ) (hn : 4 ≤ n) : ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  have hk5 : n < Q n := lt_Q n
  have hk : n ≤ 30 * Q n := by omega
  have hmem : n ∈ Icc 4 (30 * Q n) := mem_Icc.mpr ⟨hn, hk⟩
  have hin := cover_total n hmem
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp hin
  exact ⟨a, akn_sub_setA _ ha, b, akn_sub_setA _ hb, hab⟩

-- Lower bound: every element of setA is ≥ 2
lemma elt_ge2 {e : ℕ} (he : e ∈ setA) : 2 ≤ e := by
  rcases he with he | he
  · rw [Set.mem_insert_iff, Set.mem_singleton_iff] at he; omega
  · rw [Set.mem_iUnion] at he
    obtain ⟨j, hj⟩ := he
    have hp := Q_pos j
    rcases hj with (hj | hj) | hj
    · simp only [Set.mem_singleton_iff, ck] at hj; omega
    · simp only [Bk, mem_Icc] at hj; omega
    · simp only [Fk, mem_Icc] at hj; omega

-- Classification of elements relative to level k
lemma elt_class (k : ℕ) (hk : 1 ≤ k) {e : ℕ} (he : e ∈ setA) :
    e ≤ 3 * Q k ∨ e = 4 * Q k ∨ (5 * Q k ≤ e ∧ e ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ e := by
  rcases he with he | he
  · rw [Set.mem_insert_iff, Set.mem_singleton_iff] at he
    have := Q_pos k
    omega
  · rw [Set.mem_iUnion] at he
    obtain ⟨j, hj⟩ := he
    rcases lt_trichotomy j k with hlt | hje | hgt
    · -- j < k
      left
      have hb : e ≤ 15 * Q j := by
        rcases hj with (hj | hj) | hj
        · simp only [Set.mem_singleton_iff, ck] at hj; omega
        · simp only [Bk, mem_Icc] at hj; omega
        · simp only [Fk, mem_Icc] at hj; exact hj.2
      have hqj : Q j ≤ Q (k-1) := by
        unfold Q; exact Nat.pow_le_pow_right (by norm_num) (by omega)
      have hqk : Q k = 5 * Q (k-1) := by
        have h := Q_succ (k-1); rwa [Nat.sub_add_cancel hk] at h
      omega
    · -- j = k
      rw [hje] at hj
      rcases hj with (hj | hj) | hj
      · simp only [Set.mem_singleton_iff, ck] at hj; right; left; exact hj
      · simp only [Bk, mem_Icc] at hj; right; right; left; exact hj
      · simp only [Fk, mem_Icc] at hj; right; right; right; exact hj.1
    · -- k < j
      right; right; right
      have hge : 4 * Q j ≤ e := by
        rcases hj with (hj | hj) | hj
        · simp only [Set.mem_singleton_iff, ck] at hj; omega
        · simp only [Bk, mem_Icc] at hj; omega
        · simp only [Fk, mem_Icc] at hj; have := Q_pos j; omega
      have hqj : Q (k+1) ≤ Q j := by
        unfold Q; exact Nat.pow_le_pow_right (by norm_num) (by omega)
      have hqs : Q (k+1) = 5 * Q k := Q_succ k
      have := Q_pos k
      omega

-- Rigidity: sums into Jk k force one summand to be ck k
lemma rigidity_weak (k : ℕ) (hk : 1 ≤ k) {a b : ℕ}
    (ha : a ∈ setA) (hb : b ∈ setA) (hsum : a + b ∈ Jk k) :
    a = ck k ∨ b = ck k := by
  have hca := elt_class k hk ha
  have hcb := elt_class k hk hb
  have h2a := elt_ge2 ha
  have h2b := elt_ge2 hb
  rw [Jk, mem_Ico] at hsum
  have hpos := Q_pos k
  simp only [ck]
  omega

-- Gap lemma
lemma gap_lem (k : ℕ) (hk : 1 ≤ k) {T : Set ℕ} (hT : T ⊆ setA) (hc : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  rw [Set.eq_empty_iff_forall_notMem]
  intro x hx
  rw [Set.mem_inter_iff] at hx
  obtain ⟨hxJ, hxT⟩ := hx
  rw [Set.mem_add] at hxT
  obtain ⟨a, ha, b, hb, hab⟩ := hxT
  have ha' := hT ha
  have hb' := hT hb
  have hsum : a + b ∈ Jk k := by rw [hab]; exact hxJ
  rcases rigidity_weak k hk ha' hb' hsum with h | h
  · exact hc (h ▸ ha)
  · exact hc (h ▸ hb)

-- The main no-partition argument
lemma no_partition (A₁ A₂ : Set ℕ)
    (h1 : A₁ ⊆ setA) (h2 : A₂ ⊆ setA)
    (hcover : ∀ x ∈ setA, x ∈ A₁ ∨ x ∈ A₂) (hdisj : A₁ ∩ A₂ = ∅)
    (C₁ : ℕ) (hC₁ : ∀ x, ∃ m ∈ A₁ + A₁, m ∈ Icc x (x + C₁))
    (C₂ : ℕ) (hC₂ : ∀ x, ∃ m ∈ A₂ + A₂, m ∈ Icc x (x + C₂)) : False := by
  set k := C₁ + C₂ + 1 with hk_def
  have hk1 : 1 ≤ k := by omega
  have hk5 : k < Q k := lt_Q k
  have hck : ck k ∈ setA := ck_mem k
  rcases hcover (ck k) hck with hc1 | hc2
  · -- ck k ∈ A₁
    have hnotA2 : ck k ∉ A₂ := by
      intro hmem
      have hcc : ck k ∈ A₁ ∩ A₂ := ⟨hc1, hmem⟩
      rw [hdisj] at hcc; simp at hcc
    have hgap := gap_lem k hk1 h2 hnotA2
    obtain ⟨m, hmem, hmI⟩ := hC₂ (9 * Q k)
    rw [mem_Icc] at hmI
    have hmJ : m ∈ Jk k := by
      rw [Jk, mem_Ico]
      refine ⟨hmI.1, ?_⟩
      omega
    have hcontra : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hmJ, hmem⟩
    rw [hgap] at hcontra; simp at hcontra
  · -- ck k ∈ A₂
    have hnotA1 : ck k ∉ A₁ := by
      intro hmem
      have hcc : ck k ∈ A₁ ∩ A₂ := ⟨hmem, hc2⟩
      rw [hdisj] at hcc; simp at hcc
    have hgap := gap_lem k hk1 h1 hnotA1
    obtain ⟨m, hmem, hmI⟩ := hC₁ (9 * Q k)
    rw [mem_Icc] at hmI
    have hmJ : m ∈ Jk k := by
      rw [Jk, mem_Ico]
      refine ⟨hmI.1, ?_⟩
      omega
    have hcontra : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hmJ, hmem⟩
    rw [hgap] at hcontra; simp at hcontra

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, ?_, ?_⟩
  · intro n hn; exact basis n hn
  · intro A₁ A₂ hA1 hA2 hcover hdisj hsyn
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := hsyn
    exact no_partition A₁ A₂ hA1 hA2 hcover hdisj C₁ hC₁ C₂ hC₂

end Erdos741OAI

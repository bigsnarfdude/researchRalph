import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def IsAddBasis2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ (A ∪ {0}), ∃ b ∈ (A ∪ {0}), a + b = n

def Q (k : ℕ) : ℕ := 5 ^ k

lemma Q_pos (k : ℕ) : 0 < Q k := by unfold Q; exact pow_pos (by norm_num) k
lemma Q_ge_one (k : ℕ) : 1 ≤ Q k := Q_pos k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by simp [Q, pow_succ, mul_comm]

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := Icc 2 3 ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

def Akn : ℕ → Set ℕ
  | 0 => Icc 2 3
  | n + 1 => Akn n ∪ {ck n} ∪ Bk n ∪ Fk n

lemma akn_mono {m n : ℕ} (h : m ≤ n) : Akn m ⊆ Akn n := by
  induction h with
  | refl => rfl
  | @step k _ ih =>
    exact ih.trans (fun x hx => by simp only [Akn]; left; left; left; exact hx)

lemma akn_bound {k : ℕ} {x : ℕ} (hx : x ∈ Akn k) : x ≤ 3 * Q k := by
  induction k with
  | zero => simp only [Akn, mem_Icc] at hx; simp [Q]; exact hx.2
  | succ k ih =>
    simp only [Akn, mem_union, mem_singleton_iff, Bk, Fk, mem_Icc] at hx
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    rcases hx with (((h | h) | ⟨h1, h2⟩) | ⟨h1, h2⟩)
    · have := ih h; rw [hQs]; linarith
    · rw [h]; simp only [ck]; rw [hQs]; linarith
    · rw [hQs]; omega
    · rw [hQs]; omega

lemma ik_sub_akn (k : ℕ) : Icc (2 * Q k) (3 * Q k) ⊆ Akn (k + 1) := by
  cases k with
  | zero =>
    intro x hx; simp only [Q, pow_zero, mul_one] at hx
    exact Or.inl (Or.inl (Or.inl hx))
  | succ k =>
    intro x hx
    simp only [mem_Icc] at hx
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    have hx_fk : x ∈ Fk k := by
      simp only [Fk, mem_Icc]; rw [hQs] at hx; constructor <;> omega
    exact akn_mono (Nat.le_succ _)
      (Or.inr hx_fk : x ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)

lemma icc_add_icc_ge {a b c d : ℕ} (h1 : a ≤ b) (h2 : c ≤ d) :
    Icc (a + c) (b + d) ⊆ (Icc a b + Icc c d : Set ℕ) := by
  intro x hx
  simp only [mem_Icc] at hx
  simp only [Set.mem_add, mem_Icc]
  by_cases h : x ≤ a + d
  · exact ⟨a, ⟨le_refl _, h1⟩, x - a, ⟨by omega, by omega⟩, by omega⟩
  · push_neg at h
    exact ⟨x - d, ⟨by omega, by omega⟩, d, ⟨h2, le_refl _⟩, by omega⟩

-- Covers [4Qk, 6*5*Qk] by induction using pairs from {I, ck, Bk, Fk} at level k.
-- I = [2Qk, 3Qk], ck = 4Qk, Bk = [5Qk, 6Qk-1], Fk = [10Qk-1, 15Qk]
-- Pair coverage at level k: I+I=[4Q,6Q], I+ck=[6Q,7Q], I+Bk=[7Q,9Q-1],
--   ck+Bk=[9Q,10Q-1], Bk+Bk=[10Q,12Q-2], I+Fk=[12Q-1,18Q],
--   Bk+Fk=[15Q-1,21Q-1], Fk+Fk=[20Q-2,30Q]
-- The eight pairs cover [4Qk, 30Qk] using level-k pieces, all inside Akn(k+1).
lemma level_cover (k : ℕ) :
    Icc (4 * Q k) (30 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  have hcksub : ck k ∈ Akn (k + 1) :=
    (Or.inl (Or.inl (Or.inr rfl)) : ck k ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
  have hBsub : Bk k ⊆ Akn (k + 1) :=
    fun y hy => (Or.inl (Or.inr hy) : y ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
  have hFsub : Fk k ⊆ Akn (k + 1) :=
    fun y hy => (Or.inr hy : y ∈ Akn k ∪ {ck k} ∪ Bk k ∪ Fk k)
  have mI : ∀ y, 2 * Q k ≤ y → y ≤ 3 * Q k → y ∈ Akn (k + 1) :=
    fun y h1 h2 => ik_sub_akn k (mem_Icc.mpr ⟨h1, h2⟩)
  have mB : ∀ y, 5 * Q k ≤ y → y ≤ 6 * Q k - 1 → y ∈ Akn (k + 1) :=
    fun y h1 h2 => hBsub (show y ∈ Bk k from mem_Icc.mpr ⟨h1, h2⟩)
  have mF : ∀ y, 10 * Q k - 1 ≤ y → y ≤ 15 * Q k → y ∈ Akn (k + 1) :=
    fun y h1 h2 => hFsub (show y ∈ Fk k from mem_Icc.mpr ⟨h1, h2⟩)
  have mck : ∀ y, 4 * Q k ≤ y → y ≤ 4 * Q k → y ∈ Akn (k + 1) := by
    intro y h1 h2
    have hy : y = ck k := by simp only [ck]; omega
    rw [hy]; exact hcksub
  have pair : ∀ (a b c d : ℕ), a ≤ b → c ≤ d →
      (∀ y, a ≤ y → y ≤ b → y ∈ Akn (k + 1)) →
      (∀ y, c ≤ y → y ≤ d → y ∈ Akn (k + 1)) →
      Icc (a + c) (b + d) ⊆ Akn (k + 1) + Akn (k + 1) := by
    intro a b c d hab hcd hL hR
    refine (icc_add_icc_ge hab hcd).trans (Set.add_subset_add ?_ ?_)
    · intro y hy; simp only [mem_Icc] at hy; exact hL y hy.1 hy.2
    · intro y hy; simp only [mem_Icc] at hy; exact hR y hy.1 hy.2
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hlo, hhi⟩ := hx
  have hQp : 0 < Q k := Q_pos k
  by_cases h1 : x ≤ 6 * Q k
  · exact pair (2 * Q k) (3 * Q k) (2 * Q k) (3 * Q k) (by omega) (by omega) mI mI
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h2 : x ≤ 7 * Q k
  · exact pair (2 * Q k) (3 * Q k) (4 * Q k) (4 * Q k) (by omega) (by omega) mI mck
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h3 : x ≤ 9 * Q k - 1
  · exact pair (2 * Q k) (3 * Q k) (5 * Q k) (6 * Q k - 1) (by omega) (by omega) mI mB
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h4 : x ≤ 10 * Q k - 1
  · exact pair (4 * Q k) (4 * Q k) (5 * Q k) (6 * Q k - 1) (by omega) (by omega) mck mB
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h5 : x ≤ 12 * Q k - 2
  · exact pair (5 * Q k) (6 * Q k - 1) (5 * Q k) (6 * Q k - 1) (by omega) (by omega) mB mB
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h6 : x ≤ 18 * Q k
  · exact pair (2 * Q k) (3 * Q k) (10 * Q k - 1) (15 * Q k) (by omega) (by omega) mI mF
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  by_cases h7 : x ≤ 21 * Q k - 1
  · exact pair (5 * Q k) (6 * Q k - 1) (10 * Q k - 1) (15 * Q k) (by omega) (by omega) mB mF
      (mem_Icc.mpr ⟨by omega, by omega⟩)
  · exact pair (10 * Q k - 1) (15 * Q k) (10 * Q k - 1) (15 * Q k) (by omega) (by omega) mF mF
      (mem_Icc.mpr ⟨by omega, by omega⟩)

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  induction k with
  | zero =>
    intro x hx
    simp only [Q, pow_zero, mul_one, mem_Icc] at hx
    exact level_cover 0 (by simp only [Q, pow_zero, mul_one, mem_Icc]; omega)
  | succ k ih =>
    have hmono : Akn (k + 1) + Akn (k + 1) ⊆ Akn (k + 2) + Akn (k + 2) :=
      Set.add_subset_add (akn_mono (Nat.le_succ _)) (akn_mono (Nat.le_succ _))
    intro x hx
    simp only [mem_Icc] at hx
    have hQs : Q (k + 1) = 5 * Q k := Q_succ k
    have hQp : 0 < Q k := Q_pos k
    by_cases h : x ≤ 6 * Q k
    · exact hmono (ih (mem_Icc.mpr ⟨hx.1, h⟩))
    · push_neg at h
      apply hmono
      apply level_cover k
      simp only [mem_Icc]
      rw [hQs] at hx
      omega

-- Q k grows without bound: n ≤ Q n
lemma n_le_Qn (n : ℕ) : n ≤ Q n := by
  induction n with
  | zero => simp [Q]
  | succ n ih => rw [Q_succ]; linarith [Q_pos n]

-- Akn k ⊆ setA
lemma akn_sub_setA {k : ℕ} : Akn k ⊆ setA := by
  induction k with
  | zero => intro x hx; exact Or.inl hx
  | succ k ih =>
    intro x hx
    simp only [Akn, mem_union, mem_singleton_iff] at hx
    rcases hx with (((h | h) | h) | h)
    · exact ih h
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl (Set.mem_singleton_iff.mpr h))⟩)
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inr h)⟩)
    · exact Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inr h⟩)

lemma setA_covers : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  obtain ⟨a, ha, b, hb, hab⟩ := basis_lem n (mem_Icc.mpr ⟨hn, by linarith [n_le_Qn n]⟩)
  exact ⟨a, akn_sub_setA ha, b, akn_sub_setA hb, hab⟩

lemma ck_mem_setA (k : ℕ) : ck k ∈ setA :=
  Or.inr (Set.mem_iUnion.mpr ⟨k, Or.inl (Or.inl (Set.mem_singleton_iff.mpr rfl))⟩)

-- Rigidity: any sum a + b = n ∈ Jk=[9Qk,10Qk) with a,b ∈ setA must be a=ck k, b∈Bk k
-- (or vice versa). Key: elements from stage j<k are ≤3Qk, stage j>k are ≥20Qk,
-- and at stage k only the pair 4Qk + [5Qk,6Qk-1] sums into [9Qk,10Qk).
-- HINT: Use lt_trichotomy on j vs k. For the j=k case use rw [hje] at haj (not subst),
-- to keep k in scope. For Nat subtraction in hypotheses, use omega not linarith.
-- Stage-j element upper / lower bounds.
lemma stage_le (j : ℕ) {m : ℕ}
    (h : m ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ)) : m ≤ 15 * Q j := by
  simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at h
  have := Q_pos j
  rcases h with (h | ⟨_, h⟩) | ⟨_, h⟩ <;> omega

lemma stage_ge (j : ℕ) {m : ℕ}
    (h : m ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ)) : 4 * Q j ≤ m := by
  simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at h
  have := Q_pos j
  rcases h with (h | ⟨h, _⟩) | ⟨h, _⟩ <;> omega

-- Stage j < k pieces are ≤ 3Qk; stage j > k pieces are ≥ 20Qk.
lemma small_stage {k j : ℕ} (hjk : j < k) {m : ℕ}
    (h : m ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ)) : m ≤ 3 * Q k := by
  have hpow : Q (j + 1) ≤ Q k := by
    unfold Q; exact Nat.pow_le_pow_right (by norm_num) hjk
  rw [Q_succ] at hpow
  have hle := stage_le j h
  omega

lemma large_stage {k j : ℕ} (hkj : k < j) {m : ℕ}
    (h : m ∈ ({ck j} ∪ Bk j ∪ Fk j : Set ℕ)) : 20 * Q k ≤ m := by
  have hpow : Q (k + 1) ≤ Q j := by
    unfold Q; exact Nat.pow_le_pow_right (by norm_num) hkj
  rw [Q_succ] at hpow
  have hge := stage_ge j h
  omega

lemma setA_ge {m : ℕ} (h : m ∈ setA) : 2 ≤ m := by
  rcases h with hs | hst
  · simp only [mem_Icc] at hs; omega
  · simp only [mem_iUnion] at hst
    obtain ⟨j, hj⟩ := hst
    have := stage_ge j hj
    have := Q_pos j
    omega

lemma rigidity (k : ℕ) {n : ℕ} (hn : n ∈ Jk k)
    {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  simp only [Jk, mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  have hQp : 0 < Q k := Q_pos k
  have ha2 : 2 ≤ a := setA_ge ha
  have hb2 : 2 ≤ b := setA_ge hb
  have classify : ∀ m, m ∈ setA → m < 10 * Q k →
      m ≤ 3 * Q k ∨ m = 4 * Q k ∨ (5 * Q k ≤ m ∧ m ≤ 6 * Q k - 1) ∨ 10 * Q k - 1 ≤ m := by
    intro m hm hmlt
    rcases hm with hsmall | hstage
    · simp only [mem_Icc] at hsmall; left; omega
    · simp only [mem_iUnion] at hstage
      obtain ⟨j, hj⟩ := hstage
      rcases lt_trichotomy j k with hlt | hje | hgt
      · left; exact small_stage hlt hj
      · rw [hje] at hj
        simp only [mem_union, mem_singleton_iff, ck, Bk, Fk, mem_Icc] at hj
        rcases hj with (h | ⟨h1, h2⟩) | ⟨h1, h2⟩
        · right; left; exact h
        · right; right; left; exact ⟨h1, h2⟩
        · right; right; right; exact h1
      · exfalso; have := large_stage hgt hj; omega
  have hca := classify a ha (by omega)
  have hcb := classify b hb (by omega)
  rcases hca with hA | hA | hA | hA <;> rcases hcb with hB | hB | hB | hB <;>
    first
      | exact Or.inl ⟨hA, mem_Icc.mpr ⟨hB.1, hB.2⟩⟩
      | exact Or.inr ⟨hB, mem_Icc.mpr ⟨hA.1, hA.2⟩⟩
      | omega
      | (exfalso; omega)

-- Gap lemma: if ck k ∉ T (T ⊆ setA), then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [mem_inter_iff, Set.mem_add, mem_empty_iff_false, iff_false, not_and]
  intro hn
  rintro ⟨a, ha, b, hb, hab⟩
  rcases rigidity k hn (hT ha) (hT hb) hab with ⟨rfl, _⟩ | ⟨rfl, _⟩
  · exact hck ha
  · exact hck hb

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  refine ⟨setA, setA_covers, ?_⟩
  intro A₁ A₂ hA₁ hA₂ hpart hdisj ⟨⟨C₁, hC₁⟩, C₂, hC₂⟩
  obtain ⟨k, hk⟩ : ∃ k, C₁ + C₂ < Q k :=
    ⟨C₁ + C₂ + 1, lt_of_lt_of_le (Nat.lt_succ_self _) (n_le_Qn _)⟩
  have hC₁k : C₁ < Q k := by linarith
  have hC₂k : C₂ < Q k := by linarith
  rcases hpart (ck k) (ck_mem_setA k) with h₁ | h₂
  · have hck₂ : ck k ∉ A₂ := fun h => by
      have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h₁, h⟩; simp [hdisj] at hmem
    have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck₂
    obtain ⟨m, hm_sum, hm_range⟩ := hC₂ (9 * Q k)
    have hm_jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico, mem_Icc] at *
      exact ⟨hm_range.1, by linarith [hm_range.2]⟩
    have hmem : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_jk, hm_sum⟩
    simp [hgap] at hmem
  · have hck₁ : ck k ∉ A₁ := fun h => by
      have hmem : ck k ∈ A₁ ∩ A₂ := ⟨h, h₂⟩; simp [hdisj] at hmem
    have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck₁
    obtain ⟨m, hm_sum, hm_range⟩ := hC₁ (9 * Q k)
    have hm_jk : m ∈ Jk k := by
      simp only [Jk, mem_Ico, mem_Icc] at *
      exact ⟨hm_range.1, by linarith [hm_range.2]⟩
    have hmem : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_jk, hm_sum⟩
    simp [hgap] at hmem

end Erdos741OAI

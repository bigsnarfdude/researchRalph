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
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  sorry

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
lemma rigidity (k : ℕ) {n : ℕ} (hn : n ∈ Jk k)
    {a b : ℕ} (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

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

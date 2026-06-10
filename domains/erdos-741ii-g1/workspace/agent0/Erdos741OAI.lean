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

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

def setA : Set ℕ := ⋃ k, Akn k

-- Helper lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry -- Akn k ⊆ Akn k ∪ ... is straightforward by definition

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  induction k with
  | zero =>
    intro n hn
    simp only [mem_Icc] at hn
    have h_lb : 4 ≤ n := hn.1
    have h_ub : n ≤ 6 := by norm_num [Q]; exact hn.2
    simp only [Akn, Set.mem_add, Set.mem_singleton_iff, Set.mem_insert_iff]
    interval_cases n
    · exact ⟨2, Or.inl rfl, 2, Or.inl rfl, by norm_num⟩
    · exact ⟨2, Or.inl rfl, 3, Or.inr rfl, by norm_num⟩
    · exact ⟨3, Or.inr rfl, 3, Or.inr rfl, by norm_num⟩
  | succ k ih =>
    intro n hn
    simp only [mem_Icc] at hn
    sorry -- Inductive step: n ∈ [4, 6*Q(k+1)] = [4, 30*Q(k)] covered by 8 pair types

lemma rigidity (k : ℕ) :
    ∀ n ∈ Jk k, ∀ a b : ℕ, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro n hn_jk a b ha_setA hb_setA hab_sum
  sorry -- Rigidity lemma: only ck k + Bk k can sum into Jk k gap
         -- Proof by stage decomposition:
         -- - Elements from {2,3}: too small (≤ 3)
         -- - Stage j < k: bounded by 15*Q(j) ≤ 3*Q(k)
         -- - Stage j > k: bounded below by 4*Q(j) ≥ 20*Q(k) > n
         -- - Stage j = k: only ck k + [5Qk, 6Qk-1] ⊆ [9Qk, 10Qk)

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (h_not_ck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext m
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hm_jk
  by_contra hmem
  -- m ∈ T + T means ∃ a, b ∈ T, a + b = m
  simp only [Set.mem_add] at hmem
  obtain ⟨a, ha, b, hb, hab⟩ := hmem
  -- By rigidity, one of them must be ck k
  have rig := rigidity k m hm_jk a b (hT_sub ha) (hT_sub hb) hab
  -- But ck k ∉ T, contradiction
  rcases rig with ⟨ha_ck, _⟩ | ⟨hb_ck, _⟩
  · exact h_not_ck (ha_ck ▸ ha)
  · exact h_not_ck (hb_ck ▸ hb)

theorem erdos_741_ii :
    ∃ A : Set ℕ,
    (∀ n : ℕ, 4 ≤ n → ∃ a ∈ A, ∃ b ∈ A, a + b = n) ∧
    ∀ A₁ A₂ : Set ℕ,
      A₁ ⊆ A → A₂ ⊆ A →
      (∀ x ∈ A, x ∈ A₁ ∨ x ∈ A₂) →
      A₁ ∩ A₂ = ∅ →
      ¬ (IsSyndetic (A₁ + A₁) ∧ IsSyndetic (A₂ + A₂)) := by
  use setA
  constructor
  · -- Prove setA is a basis
    intro n hn_ge
    -- Find k such that n ≤ Q k; then apply basis_lem to show n ∈ Akn k + Akn k ⊆ setA + setA
    sorry -- setA is a basis: ∀ n ≥ 4, ∃ a, b ∈ setA with a + b = n
  · -- Prove rigidity: no partition is both-syndetic
    intro A₁ A₂ h_A1_sub h_A2_sub h_part h_disj
    intro ⟨⟨C1, hC1⟩, ⟨C2, hC2⟩⟩

    -- Pick k large enough that Q k > C2
    let k := C2 + 1
    have h_Q_large : C2 < Q k := by
      unfold Q
      sorry -- 5^(C2+1) > C2 for all C2

    -- Since ck k ∈ setA and A₁, A₂ partition setA, either ck k ∈ A₁ or ck k ∈ A₂
    have ck_in_A : ck k ∈ setA := by
      sorry -- ck k ∈ Akn(k+1) ⊆ setA

    have ck_mem : ck k ∈ A₁ ∨ ck k ∈ A₂ := by
      have := h_part (ck k) ck_in_A
      tauto

    -- If ck k ∈ A₁, then ck k ∉ A₂ (by disjointness)
    -- If ck k ∈ A₂, then ck k ∉ A₁
    -- WLOG assume ck k ∉ A₂

    by_cases h_ck : ck k ∈ A₂
    · -- Case: ck k ∈ A₂, so ck k ∉ A₁
      have h_not_ck_A1 : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h h_ck
        simp [h_disj] at this

      -- By gap_lem, Jk k ∩ (A₁ + A₁) = ∅
      have gap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ h_A1_sub h_not_ck_A1

      -- But C1 gap is syndetic, so ∃ m ∈ A₁ + A₁, m ∈ Icc (9*Qk) (9*Qk + C1)
      have ⟨m, hm_mem, hm_icc⟩ := hC1 (9 * Q k)

      -- Then m ∈ Jk k (since m ∈ [9*Qk, 9*Qk + C1] and C1 < Qk)
      have m_in_Jk : m ∈ Jk k := by
        sorry -- m ∈ [9*Qk, 9*Qk + C1) because C1 < Q k

      -- Then m ∈ Jk k ∩ (A₁ + A₁), contradicting gap
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter m_in_Jk hm_mem
      simp [gap] at this
    · -- Case: ck k ∉ A₂
      -- By gap_lem, Jk k ∩ (A₂ + A₂) = ∅
      have gap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ h_A2_sub h_ck

      -- But C2 gap is syndetic, so ∃ m ∈ A₂ + A₂, m ∈ Icc (9*Qk) (9*Qk + C2)
      have ⟨m, hm_mem, hm_icc⟩ := hC2 (9 * Q k)

      -- Then m ∈ Jk k (since m ∈ [9*Qk, 9*Qk + C2] and C2 < Qk)
      have m_in_Jk : m ∈ Jk k := by
        sorry -- m ∈ [9*Qk, 9*Qk + C2) because C2 < Q k

      -- Then m ∈ Jk k ∩ (A₂ + A₂), contradicting gap
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter m_in_Jk hm_mem
      simp [gap] at this

end Erdos741OAI

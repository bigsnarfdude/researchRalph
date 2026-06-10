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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Basic lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp only [pow_succ, mul_comm 5]

-- Helper to relate Akn to setA
lemma akn_mono (k : ℕ) : Akn k ⊆ setA := by
  sorry

-- Basis lemma: Akn covers all sums up to 6 * Q k
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [Set.mem_Icc] at hx
  obtain ⟨h_lo, h_hi⟩ := hx
  -- We use 2-induction on k to show this
  -- For k = 0: need to cover [4, 6] with sums from {2, 3}
  -- For k+1: inductively use coverage up to k, plus new elements
  sorry

-- Rigidity lemma: elements in Jk k have restricted sum decompositions
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) :
    ∀ a b, a ∈ setA → b ∈ setA → a + b = n →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intro a b ha hb hab_eq
  -- This is a complex lemma requiring careful stage analysis
  -- For now, we assert the result
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k ∩ (T + T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  by_contra h
  have : (Jk k ∩ (T + T)).Nonempty := by
    simp only [Set.nonempty_iff_ne_empty]
    exact h
  obtain ⟨n, hn_j, hn_add⟩ := this
  simp only [Set.mem_add] at hn_add
  obtain ⟨a, ha, b, hb, hab⟩ := hn_add
  -- By rigidity_lem, either a = ck k or b = ck k
  have hrig := rigidity_lem k n hn_j a b (hT ha) (hT hb) hab
  rcases hrig with ⟨hck_a, hb_mem⟩ | ⟨hck_b, ha_mem⟩
  · exact hck (hck_a ▸ ha)
  · exact hck (hck_b ▸ hb)

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
  · intro n hn
    -- Find k such that n ≤ 6 * Q k
    have : ∃ k, n ≤ 6 * Q k := by
      sorry  -- Q grows without bound
    obtain ⟨k, hk_bound⟩ := this
    -- Apply basis_lem for this k
    have : n ∈ Icc 4 (6 * Q k) := by
      simp only [Set.mem_Icc]
      exact ⟨hn, hk_bound⟩
    have : n ∈ Akn (k + 1) + Akn (k + 1) := basis_lem k this
    -- Elements of Akn are in setA by akn_mono
    simp only [Set.mem_add] at this
    obtain ⟨a, ha, b, hb, hab⟩ := this
    exact ⟨a, akn_mono (k + 1) ha, b, akn_mono (k + 1) hb, hab⟩
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    -- Choose k large enough so that Q k > max(C₁, C₂)
    -- We use k = 1 + max(C₁, C₂), which works since 5^(1 + n) > n for all n
    let k : ℕ := max C₁ C₂ + 1
    have hk_bound : max C₁ C₂ < Q k := by
      unfold Q
      sorry  -- 5^(max C₁ C₂ + 1) > max C₁ C₂
    have hck : ck k ∈ setA := by
      sorry
    -- Since ck k ∈ setA and A = A₁ ⊔ A₂, we have ck k ∈ A₁ or ck k ∈ A₂
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck
    rcases this with hck₁ | hck₂
    · -- ck k ∈ A₁, so ck k ∉ A₂
      have hck₂_not : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck₁ h
        simp [hdisj] at this
      -- By gap_lem, Jk k ∩ (A₂ + A₂) = ∅
      have hgap := gap_lem k A₂ hA₂ hck₂_not
      -- But A₂ + A₂ is syndetic with bound C₂, so it hits [9*Qk, 9*Qk + C₂] ⊆ Jk k
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := by
        obtain ⟨m, hm_mem, hm_icc⟩ := hC₂ (9 * Q k)
        exact ⟨m, hm_mem, hm_icc⟩
      obtain ⟨m, hm_add, hm_icc⟩ := this
      -- m ∈ Icc (9*Qk) (9*Qk + C₂) ⊆ Ico (9*Qk) (10*Qk) = Jk k
      have hm_jk : m ∈ Jk k := by
        simp only [Set.mem_Icc] at hm_icc
        unfold Jk
        simp only [Set.mem_Ico]
        obtain ⟨h1, h2⟩ := hm_icc
        constructor
        · exact h1
        · have : C₂ < Q k := by omega
          omega
      -- But m ∈ Jk k ∩ (A₂ + A₂), contradicting hgap
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hm_jk hm_add
      simp [hgap] at this
    · -- ck k ∈ A₂, so ck k ∉ A₁
      have hck₁_not : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h hck₂
        simp [hdisj] at this
      -- By gap_lem, Jk k ∩ (A₁ + A₁) = ∅
      have hgap := gap_lem k A₁ hA₁ hck₁_not
      -- But A₁ + A₁ is syndetic with bound C₁, so it hits [9*Qk, 9*Qk + C₁] ⊆ Jk k
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := by
        obtain ⟨m, hm_mem, hm_icc⟩ := hC₁ (9 * Q k)
        exact ⟨m, hm_mem, hm_icc⟩
      obtain ⟨m, hm_add, hm_icc⟩ := this
      -- m ∈ Icc (9*Qk) (9*Qk + C₁) ⊆ Ico (9*Qk) (10*Qk) = Jk k
      have hm_jk : m ∈ Jk k := by
        simp only [Set.mem_Icc] at hm_icc
        unfold Jk
        simp only [Set.mem_Ico]
        obtain ⟨h1, h2⟩ := hm_icc
        constructor
        · exact h1
        · have : C₁ < Q k := by omega
          omega
      -- But m ∈ Jk k ∩ (A₁ + A₁), contradicting hgap
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter hm_jk hm_add
      simp [hgap] at this

end Erdos741OAI

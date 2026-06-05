import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q (k : ℕ) : ℕ := 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k

def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)

def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)

def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k : ℕ, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : (0 : ℕ) < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ, mul_comm]

lemma akn_mono {k₁ k₂ : ℕ} (h : k₁ ≤ k₂) : Akn k₁ ⊆ Akn k₂ := by
  sorry

lemma basis_lem (k : ℕ) (x : ℕ) (hx : 4 ≤ x ∧ x ≤ 6 * Q k) :
    ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  obtain ⟨hx_lo, hx_hi⟩ := hx
  sorry

lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (ha : ∃ a b : ℕ, a ∈ setA ∧ b ∈ setA ∧ a + b = n) :
    ∃ a ∈ setA, ∃ b ∈ setA, (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

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
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    by_contra h_contra
    -- Assume both parts are syndetic
    push_neg at h_contra
    -- Case split: ck 0 must be in A₁ or A₂
    have hck0_in : ck 0 ∈ A₁ ∨ ck 0 ∈ A₂ := by
      have : ck 0 ∈ setA := by sorry
      exact hpart (ck 0) this
    cases hck0_in with
    | inl hck0_A1 =>
      -- ck 0 ∈ A₁, so by gap_lem, Jk 0 ∩ (A₂ + A₂) = ∅
      have hck0_notA2 : ck 0 ∉ A₂ := by
        intro h
        have : ck 0 ∈ A₁ ∩ A₂ := ⟨hck0_A1, h⟩
        simp [hdisj] at this
      have gap := gap_lem 0 A₂ hA₂ hck0_notA2
      -- gap says: Jk 0 ∩ (A₂ + A₂) = ∅
      -- But hC₂ says: ∀ x, ∃ m ∈ A₂ + A₂, m ∈ Icc x (x + C₂)
      -- Take x = 9 * Q 0 to get a contradiction
      have := hC₂ (9 * Q 0)
      obtain ⟨m, hm_mem, hm_icc⟩ := this
      -- m ∈ Icc (9*Q 0) (9*Q 0 + C₂), so m ∈ Jk 0
      have hm_jk : m ∈ Jk 0 := by
        unfold Jk Ico at *
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_icc
        constructor
        · exact hlo
        · omega
      -- But m ∈ Jk 0 ∩ (A₂ + A₂)
      have : m ∈ Jk 0 ∩ (A₂ + A₂) := ⟨hm_jk, hm_mem⟩
      -- This contradicts gap
      simp [gap] at this
    | inr hck0_A2 =>
      -- ck 0 ∈ A₂, so by gap_lem, Jk 0 ∩ (A₁ + A₁) = ∅
      have hck0_notA1 : ck 0 ∉ A₁ := by
        intro h
        have : ck 0 ∈ A₁ ∩ A₂ := ⟨h, hck0_A2⟩
        simp [hdisj] at this
      have gap := gap_lem 0 A₁ hA₁ hck0_notA1
      sorry

end Erdos741OAI

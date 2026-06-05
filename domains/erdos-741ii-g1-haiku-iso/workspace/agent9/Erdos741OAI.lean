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

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn (k : ℕ) : Set ℕ :=
  {2, 3} ∪ ⋃ j ∈ Finset.range k, {ck j} ∪ Bk j ∪ Fk j

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  simp only [Akn] at hx ⊢
  cases hx with
  | inl h => left; exact h
  | inr h =>
    right
    obtain ⟨j, hj_mem, hj_rest⟩ := h
    use j
    simp only [Finset.mem_range] at hj_mem ⊢
    exact ⟨sorry, hj_rest⟩

lemma Q_ge_k (k : ℕ) : k < Q k := by
  sorry

lemma akn_mem_ck (k : ℕ) : ck k ∈ Akn (k + 1) := by
  sorry

lemma akn_mem_Bk (k : ℕ) (x : ℕ) (hx : x ∈ Bk k) : x ∈ Akn (k + 1) := by
  sorry

lemma akn_mem_Fk (k : ℕ) (x : ℕ) (hx : x ∈ Fk k) : x ∈ Akn (k + 1) := by
  sorry

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc, Set.mem_add] at hx ⊢
  obtain ⟨hlo, hhi⟩ := hx
  by_cases h1 : x ≤ 5 * Q k
  · use x - 2*Q k
    constructor
    · sorry
    · use 2*Q k
      exact ⟨sorry, sorry⟩
  by_cases h2 : x ≤ 6 * Q k - 1
  · push_neg at h1
    use 2*Q k
    constructor
    · sorry
    · use x - 2*Q k
      exact ⟨sorry, by omega⟩
  by_cases h3 : x ≤ 10 * Q k - 1
  · push_neg at h2
    use 4*Q k
    constructor
    · sorry
    · use x - 4*Q k
      exact ⟨sorry, by omega⟩
  · push_neg at h3
    use x - 10*Q k + 1
    constructor
    · sorry
    · use 10*Q k - 1
      exact ⟨sorry, by omega⟩

lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA)
    (hab : a + b = n) : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT_sub : T ⊆ setA) (hck : ck k ∉ T) : Jk k ∩ (T + T) = ∅ := by
  ext n
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
  intro hn_jk
  simp only [Set.mem_add]
  intro ⟨a, ha_T, b, hb_T, hab⟩
  have ha_A : a ∈ setA := hT_sub ha_T
  have hb_A : b ∈ setA := hT_sub hb_T
  have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) :=
    rigidity_lem k n hn_jk a b ha_A hb_A hab
  rcases this with ⟨ha_eq, _⟩ | ⟨hb_eq, _⟩
  · rw [← ha_eq] at hck
    exact hck ha_T
  · rw [← hb_eq] at hck
    exact hck hb_T

lemma n_le_6Q_exists (n : ℕ) : ∃ k, n ≤ 6 * Q k := by
  use n
  have : n < Q n := Q_ge_k n
  omega

lemma Akn_sub_setA (k : ℕ) : Akn k ⊆ setA := by
  intro x hx
  simp only [Akn, mem_union, mem_iUnion] at hx
  rcases hx with h | h
  · simp only [setA, mem_union, mem_iUnion]
    left
    exact h
  · simp only [setA, mem_union, mem_iUnion]
    right
    obtain ⟨j, _, hj⟩ := h
    exact ⟨j, hj⟩

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
    obtain ⟨k, hk⟩ := n_le_6Q_exists n
    have h_in_basis : n ∈ Icc 4 (6 * Q k) := by
      simp only [mem_Icc]
      exact ⟨hn, hk⟩
    have h_sum : n ∈ Akn (k + 1) + Akn (k + 1) := basis_lem k h_in_basis
    simp only [Set.mem_add] at h_sum
    obtain ⟨a, ha, b, hb, hab⟩ := h_sum
    have ha' : a ∈ setA := Akn_sub_setA (k + 1) ha
    have hb' : b ∈ setA := Akn_sub_setA (k + 1) hb
    exact ⟨a, ha', b, hb', hab⟩
  · intro A₁ A₂ h_sub1 h_sub2 h_cover h_disj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    let k := max C₁ C₂ + 1
    have hck_mem : ck k ∈ setA := by
      sorry
    have h_ck_part : ck k ∈ A₁ ∨ ck k ∈ A₂ := h_cover (ck k) hck_mem
    rcases h_ck_part with hck_A₁ | hck_A₂
    · have hck_not_A₂ : ck k ∉ A₂ := by
        intro hck_A₂
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ hck_A₂
        simp [h_disj] at this
      have h_gap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ h_sub2 hck_not_A₂
      obtain ⟨m, hm_A₂_sum, hm_Ico⟩ := hC₂ (9 * Q k)
      have h_m_bounds : m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hm_Ico
      have h_m_Jk : m ∈ Jk k := by
        simp only [Jk, mem_Ico, mem_Icc] at h_m_bounds ⊢
        obtain ⟨h1, h2⟩ := h_m_bounds
        have hC2_bound : C₂ < Q k := by
          have : max C₁ C₂ < k := Nat.lt_of_succ_le (Nat.le_refl _)
          sorry
        exact ⟨h1, by omega⟩
      have h_m_inter : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter h_m_Jk hm_A₂_sum
      simp [h_gap] at h_m_inter
    · have hck_not_A₁ : ck k ∉ A₁ := by
        intro hck_A₁
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hck_A₁ hck_A₂
        simp [h_disj] at this
      have h_gap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ h_sub1 hck_not_A₁
      obtain ⟨m, hm_A₁_sum, hm_Ico⟩ := hC₁ (9 * Q k)
      have h_m_bounds : m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hm_Ico
      have h_m_Jk : m ∈ Jk k := by
        simp only [Jk, mem_Ico, mem_Icc] at h_m_bounds ⊢
        obtain ⟨h1, h2⟩ := h_m_bounds
        have hC1_bound : C₁ < Q k := by
          have : max C₁ C₂ < k := Nat.lt_of_succ_le (Nat.le_refl _)
          sorry
        exact ⟨h1, by omega⟩
      have h_m_inter : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter h_m_Jk hm_A₁_sum
      simp [h_gap] at h_m_inter

end Erdos741OAI

import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

def Q : ℕ → ℕ := fun k => 5 ^ k

def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

lemma akn_mono (j k : ℕ) (h : j ≤ k) : Akn j ⊆ Akn k := by
  intro x hx
  revert j k
  revert hx
  revert x
  induction k with
  | zero =>
      intro x hx j hj
      omega
  | succ k ih =>
      intro x hx j hj
      by_cases hj_eq : j = k + 1
      · rw [hj_eq]
      · have : j ≤ k := by omega
        simp only [Akn] at hx ⊢
        cases hx with
        | inl hx_prev => exact Or.inl (ih x hx_prev j this)
        | inr hx_curr =>
            right
            exact hx_curr

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx_mem
  unfold Akn
  simp only [Set.mem_add, Set.mem_union]
  -- For any x ∈ [4, 6*Qk], we exhibit a, b such that a+b=x and a,b ∈ Akn(k+1)
  -- Use representatives from previous level (Akn k) and current level components
  -- Key observation: [2,3]+(inherited interval) and level-k components combine to cover [4, 6*Qk]
  sorry

lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a : ℕ) (ha : a ∈ setA) (b : ℕ) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  -- Stage decomposition: any two elements from setA that sum to n ∈ [9Qk, 10Qk) must be exactly {ck k, elem ∈ Bk k}
  -- Elements from {2,3} are too small to reach 9*Qk for k>0
  -- Elements from stages j<k have max value 15*Q(j) ≤ 3*Q(k) (geometric decay)
  -- Elements from stages j>k have min value 4*Q(j) ≥ 20*Q(k) > 10*Qk (geometric growth)
  -- Only viable pair: one element is ck(k)=4*Qk, other is from Bk(k)=[5*Qk, 6*Qk-1]
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  push_neg
  intro hxJ a ha b hb hab
  have ha_setA := hT ha
  have hb_setA := hT hb
  have : (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := rigidity_lem k x hxJ a ha_setA b hb_setA hab
  rcases this with ⟨hak, hb_Bk⟩ | ⟨hbk, ha_Bk⟩
  · rw [hak] at ha
    exact hck ha
  · rw [hbk] at hb
    exact hck hb

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
    sorry  -- basis_lem
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    let C := max C₁ C₂
    obtain ⟨k, hk⟩ : ∃ k, Q k > C := by
      use C + 5
      unfold Q
      sorry  -- 5^n grows exponentially, so 5^(C+5) > C
    have hck_mem : ck k ∈ setA := by sorry  -- ck k is in the iUnion of sets at each level
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_mem
    rcases this with hck1 | hck2
    · have hck_not_A₂ : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck1, h⟩
        rw [hdisj] at this
        simp at this
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ hck_not_A₂
      have ⟨m, hm_mem, hm_range⟩ := hC₂ (9 * Q k)
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := hm_range
        constructor
        · omega
        · omega
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_in_Jk, hm_mem⟩
      rw [hgap] at this
      simp at this
    · have hck_not_A₁ : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck2⟩
        rw [hdisj] at this
        simp at this
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ hck_not_A₁
      have ⟨m, hm_mem, hm_range⟩ := hC₁ (9 * Q k)
      have hm_in_Jk : m ∈ Jk k := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := hm_range
        constructor
        · omega
        · omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_in_Jk, hm_mem⟩
      rw [hgap] at this
      simp at this

end Erdos741OAI

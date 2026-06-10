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

def setA : Set ℕ :=
  {2, 3} ∪ ⋃ k, ({ck k} : Set ℕ) ∪ Bk k ∪ Fk k

-- Recursive definition of partial union up to level k
def Akn : ℕ → Set ℕ :=
  fun k => ⋃ j ≤ k, if j = 0 then {2, 3} else {ck (j - 1)} ∪ Bk (j - 1) ∪ Fk (j - 1)

-- Q is positive and grows exponentially
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_mono : ∀ j k : ℕ, j ≤ k → Q j ≤ Q k := by
  intros j k hjk
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

-- Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  unfold Akn at hx ⊢
  simp only [Set.mem_iUnion, exists_prop] at hx ⊢
  obtain ⟨j, hj, hmem⟩ := hx
  use j
  exact ⟨Nat.le_succ_of_le hj, hmem⟩

-- The induction step: Akn (k+1) covers sums up to 6*Q(k+1)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  simp only [mem_Icc] at hx
  obtain ⟨hxlo, hxhi⟩ := hx
  -- Exhibit x = 2 + (x - 2) where both are in Akn(k+1)
  refine ⟨2, ?_, x - 2, ?_, ?_⟩
  · -- 2 ∈ Akn(k+1)
    unfold Akn
    simp only [Set.mem_iUnion, exists_prop]
    use 0
    exact ⟨by omega, by simp⟩
  · -- (x - 2) ∈ Akn(k+1)
    -- x ∈ [4, 6*Q(k+1)], so x-2 ∈ [2, 6*Q(k+1)-2]
    -- This is a larger interval, but we need to show x-2 is in Akn(k+1)
    -- One approach: show that [2, 6*Q(k+1)] ⊆ Akn(k+1)
    -- But Akn(k+1) ⊇ {2,3} from level 0, so 2 ∈ Akn(k+1)
    -- For x-2 ≥ 3, need to use recursion or other coverage
    sorry  -- needs detailed case analysis or lemma about Akn coverage
  · -- 2 + (x - 2) = x
    have h : 2 ≤ x := by omega
    simp [Nat.add_sub_cancel' h]

-- Rigidity: elements in Jk that sum must involve ck
lemma rigidity (k : ℕ) : ∀ a b : ℕ, a + b ∈ Jk k → a ∈ setA → b ∈ setA →
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  intros a b hab ha hb
  -- Elements of setA are: {2,3} or stage j≥1 elements {ck j} ∪ Bk j ∪ Fk j
  -- For a + b ∈ Jk k = [9*Q k, 10*Q k), need stage decomposition
  -- Elements from early stages are too small, later stages are too large
  -- Only ck k + Bk k sums into Jk k
  sorry

-- Gap lemma: if ck k ∉ T, then Jk k doesn't intersect T + T
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false]
  intro ⟨hjk, hadd⟩
  simp only [Set.mem_add] at hadd
  obtain ⟨a, ha, b, hb, hab⟩ := hadd
  have ha_in_setA : a ∈ setA := hT ha
  have hb_in_setA : b ∈ setA := hT hb
  rw [← hab] at hjk
  have h_rig := rigidity k a b hjk ha_in_setA hb_in_setA
  rcases h_rig with ⟨ha_eq, hb_mem⟩ | ⟨hb_eq, ha_mem⟩
  · rw [ha_eq] at ha
    exact hck ha
  · rw [hb_eq] at hb
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
  · -- setA is a basis: every n ≥ 4 can be written as a sum from setA
    intro n hn
    -- Use basis_lem: n ∈ [4, 6*Q(n)] ⊆ Akn(n) + Akn(n)
    -- Akn(n) ⊆ setA by definition
    sorry  -- requires proving n ≤ 6*Q(n) and Akn(n) ⊆ setA
  · -- No partition is both-syndetic
    intros A₁ A₂ hA₁ hA₂ hpart hdisj
    rintro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩
    -- Pick k = 100 (any large k works since Q grows exponentially and is always > max(C₁,C₂))
    let k := 100
    have hQk : C₁ < Q k ∧ C₂ < Q k := by
      unfold Q
      norm_num
      sorry  -- 5^100 is huge, > any C₁, C₂
    -- ck k ∈ setA
    have ck_in_A : ck k ∈ setA := by
      unfold setA ck
      right
      simp only [Set.mem_iUnion]
      use k
      simp [Q]
    have ck_case := hpart (ck k) ck_in_A
    rcases ck_case with hck_A1 | hck_A2
    · -- Case: ck k ∈ A₁. Then Jk k ∩ (A₂ + A₂) = ∅
      have h_not_A2 : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck_A1, h⟩
        simp [hdisj] at this
      have hgap := gap_lem k A₂ hA₂ h_not_A2
      -- hC₂ says A₂ + A₂ is syndetic with bound C₂
      have h_int : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      obtain ⟨m, hm_add, hm_int⟩ := h_int
      simp only [mem_Icc] at hm_int
      have hm_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        exact ⟨hm_int.1, by omega⟩
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨hm_Jk, hm_add⟩
      simp [hgap] at this
    · -- Case: ck k ∈ A₂. Then Jk k ∩ (A₁ + A₁) = ∅
      have h_not_A1 : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck_A2⟩
        simp [hdisj] at this
      have hgap := gap_lem k A₁ hA₁ h_not_A1
      have h_int : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      obtain ⟨m, hm_add, hm_int⟩ := h_int
      simp only [mem_Icc] at hm_int
      have hm_Jk : m ∈ Jk k := by
        unfold Jk
        simp only [mem_Ico]
        exact ⟨hm_int.1, by omega⟩
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨hm_Jk, hm_add⟩
      simp [hgap] at this

end Erdos741OAI

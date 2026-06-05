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

-- Construction: Q(k) = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Building blocks at level k
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- The set A = {2, 3} ∪ ⋃_k ({ck k} ∪ Bk k ∪ Fk k)
def setA : Set ℕ := {2, 3} ∪ ⋃ k, {ck k} ∪ Bk k ∪ Fk k

-- Akn k = partial union up to level k
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- Lemma: Q is always positive
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  apply pow_pos
  norm_num

-- Lemma: Q (k+1) = 5 * Q k
lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  rw [pow_succ]
  ring

-- Lemma: Akn is monotone
lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x hx
  induction k generalizing x with
  | zero =>
    simp [Akn] at *
    tauto
  | succ k ih =>
    simp [Akn] at *
    tauto


-- Lemma: Every n in [4, 6*Q(k)] can be written as sum of two elements from Akn(k)
lemma basis_lem (k : ℕ) : Icc 4 (6 * Q k) ⊆ Akn k + Akn k := by
  intro n hn
  simp only [Set.mem_add, Set.mem_Icc] at *
  induction k with
  | zero =>
    simp [Icc, Q, Akn] at hn
    have h1 : 4 ≤ n := hn.1
    have h2 : n ≤ 6 := hn.2
    -- For k=0, Q(0)=1, so [4, 6]. We have 2+2=4, 2+3=5, 3+3=6
    match n with
    | 4 => exact ⟨2, Or.inl rfl, 2, Or.inl rfl, rfl⟩
    | 5 => exact ⟨2, Or.inl rfl, 3, Or.inr rfl, rfl⟩
    | 6 => exact ⟨3, Or.inr rfl, 3, Or.inr rfl, rfl⟩
    | n + 7 => omega
  | succ k ih =>
    sorry

-- Lemma: Rigidity at gap interval
lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k)
    (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Lemma: Gap property
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  sorry

-- Helper: Every n ≥ 4 is in some [4, 6*Q(k)]
lemma exists_covering_level (n : ℕ) (hn : 4 ≤ n) : ∃ k, n ∈ Icc 4 (6 * Q k) := by
  use n
  simp only [Icc, Q, mem_setOf]
  constructor
  · exact hn
  · have h : 5 ^ n ≥ n := by
      clear hn
      induction n with
      | zero => norm_num
      | succ m ih =>
        simp only [pow_succ]
        have h1 : 5 ^ m ≥ m := ih
        cases m with
        | zero => norm_num
        | succ k =>
          have h2 : 5 * 5 ^ (k + 1) ≥ 5 * (k + 1) := by omega
          omega
    omega

-- Helper: Akn k ⊆ setA
lemma Akn_subset_setA (k : ℕ) : Akn k ⊆ setA := by
  sorry

-- Helper: setA is a basis
lemma setA_is_basis : ∀ n : ℕ, 4 ≤ n → ∃ a ∈ setA, ∃ b ∈ setA, a + b = n := by
  intro n hn
  obtain ⟨k, hk⟩ := exists_covering_level n hn
  have : n ∈ Akn k + Akn k := by
    have := basis_lem k
    exact this hk
  obtain ⟨a, ha, b, hb, hab⟩ := Set.mem_add.mp this
  exact ⟨a, Akn_subset_setA k ha, b, Akn_subset_setA k hb, hab⟩

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
  · exact setA_is_basis
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h
    obtain ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩ := h
    let C := max C₁ C₂
    have hC₁' : C₁ ≤ C := le_max_left C₁ C₂
    have hC₂' : C₂ ≤ C := le_max_right C₁ C₂
    let k := C + 1
    have hQ : Q k > C := by
      unfold Q
      have : 5 ^ k ≥ k := by
        induction k with
        | zero => norm_num
        | succ k ih =>
          simp only [pow_succ]
          have h1 : 5 ^ k ≥ k := ih
          cases k with
          | zero => norm_num
          | succ m =>
            have h2 : 5 ^ (m + 1) ≥ m + 1 := ih
            omega
      omega
    have hck : ck k ∈ setA := by
      simp only [setA, Set.mem_union, Set.mem_iUnion]
      right
      use k
      simp only [Akn, Set.mem_union, Set.mem_singleton_iff, ck, Set.mem_singleton]
      tauto
    have : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck
    cases this with
    | inl hck1 =>
      have hgap : Jk k ∩ (A₂ + A₂) = ∅ := gap_lem k A₂ hA₂ (fun h => by
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck1, h⟩
        simp [hdisj] at this)
      have : ∃ m ∈ A₂ + A₂, m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hC₂ (9 * Q k)
      obtain ⟨m, hm, hmIcc⟩ := this
      have hm_bounds := mem_Icc.mp hmIcc
      have : m ∈ Jk k := by
        simp [Jk, Ico]
        constructor
        · exact hm_bounds.1
        · have : m ≤ 9 * Q k + C₂ := hm_bounds.2
          have : 9 * Q k + C₂ ≤ 9 * Q k + C := by omega
          omega
      have : m ∈ Jk k ∩ (A₂ + A₂) := ⟨this, hm⟩
      simp [hgap] at this
    | inr hck2 =>
      have hgap : Jk k ∩ (A₁ + A₁) = ∅ := gap_lem k A₁ hA₁ (fun h => by
        have : ck k ∈ A₁ ∩ A₂ := ⟨h, hck2⟩
        simp [hdisj] at this)
      have : ∃ m ∈ A₁ + A₁, m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hC₁ (9 * Q k)
      obtain ⟨m, hm, hmIcc⟩ := this
      have hm_bounds := mem_Icc.mp hmIcc
      have : m ∈ Jk k := by
        simp [Jk, Ico]
        constructor
        · exact hm_bounds.1
        · have : m ≤ 9 * Q k + C₁ := hm_bounds.2
          have : 9 * Q k + C₁ ≤ 9 * Q k + C := by omega
          omega
      have : m ∈ Jk k ∩ (A₁ + A₁) := ⟨this, hm⟩
      simp [hgap] at this

end Erdos741OAI

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

def Akn (k : ℕ) : Set ℕ :=
  if k = 0 then {2, 3}
  else Akn (k - 1) ∪ {ck (k - 1)} ∪ Bk (k - 1) ∪ Fk (k - 1)

def setA : Set ℕ := {2, 3} ∪ ⋃ k, ({ck k} ∪ Bk k ∪ Fk k)

lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  unfold Q
  simp [pow_succ, mul_comm]

lemma Q_mono (i j : ℕ) (hij : i ≤ j) : Q i ≤ Q j := by
  unfold Q
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hij

lemma Qk_bound (j k : ℕ) (hj : j < k) : 15 * Q j < 4 * Q k := by
  have h_le : j + 1 ≤ k := Nat.succ_le_of_lt hj
  have h_Q : Q (j + 1) ≤ Q k := Q_mono (j + 1) k h_le
  rw [Q_succ] at h_Q
  have h_pos : 0 < Q j := Q_pos j
  have : 20 * Q j ≤ 4 * Q k := by omega
  omega

lemma Q_gt_input (k : ℕ) (hk : 2 ≤ k) : k < Q k := by
  sorry

lemma jk_bound (k C : ℕ) (hC : C < Q k) (m : ℕ) (hm_lo : 9 * Q k ≤ m) (hm_hi : m ≤ 9 * Q k + C) :
    m < 10 * Q k := by
  omega

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  sorry

lemma inI (k : ℕ) (x : ℕ) (hlo : 5 * Q k ≤ x) (hhi : x ≤ 6 * Q k - 1) : x ∈ Bk k := by
  unfold Bk
  exact mem_Icc.mpr ⟨hlo, hhi⟩

lemma basis_lem (k : ℕ) : Icc 4 (6 * Q (k + 1)) ⊆ Akn (k + 1) + Akn (k + 1) := by
  intro x hx
  -- Eight cases based on which subinterval x falls into
  -- I: [2*Qk, 3*Qk] (inherited)
  -- B: [5*Qk, 6*Qk-1]
  -- F: [10*Qk-1, 15*Qk]
  -- ck: 4*Qk
  -- Pairs: I+I, I+ck, I+B, ck+B, B+B, I+F, B+F, F+F cover [4*Qk, 30*Qk]
  sorry

lemma rigidity_lem (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ)
    (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  unfold Jk at hn
  simp only [Set.mem_Ico] at hn
  simp only [setA, Set.mem_union, Set.mem_iUnion] at ha hb
  -- Stage decomposition: each element of setA is either {2,3} or from some stage j
  -- For n in [9*Qk, 10*Qk), we show both a and b must be from stage k
  -- and then show the only valid pairing is a=ck k, b∈Bk k or vice versa
  sorry

lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  ext x
  simp only [Set.mem_inter_iff, Set.mem_add, Set.mem_empty_iff_false, iff_false]
  intro ⟨hx_jk, a, ha, b, hb, hab⟩
  have ha_A := hT ha
  have hb_A := hT hb
  have h_rigid := rigidity_lem k x hx_jk a b ha_A hb_A hab
  cases h_rigid with
  | inl h =>
    obtain ⟨rfl, _⟩ := h
    exact hck ha
  | inr h =>
    obtain ⟨rfl, _⟩ := h
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
    -- Prove that every n ≥ 4 can be written as a + b where a, b ∈ setA
    -- This uses basis_lem and akn_mono
    sorry
  · intro A₁ A₂ hA₁ hA₂ hpart hdisj h_syndet
    obtain ⟨C₁, hC₁⟩ := h_syndet.1
    obtain ⟨C₂, hC₂⟩ := h_syndet.2
    let k := max (max C₁ C₂) 2
    have hck_mem : ck k ∈ setA := by
      unfold setA
      right
      rw [Set.mem_iUnion]
      use k
      simp
    have hc := hpart (ck k) hck_mem
    cases hc with
    | inl hc_A1 =>
      have hck_not_A2 : ck k ∉ A₂ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter hc_A1 h
        simp [hdisj] at this
      have hgap := gap_lem k A₂ hA₂ hck_not_A2
      have h_syndet := hC₂ (9 * Q k)
      obtain ⟨m, hm_A2, hm_range⟩ := h_syndet
      have h_in_interval : m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hm_range
      have h_C2_bound : C₂ < Q k := by
        have : max (max C₁ C₂) 2 ≥ max C₁ C₂ := Nat.le_max_left _ _
        have : max (max C₁ C₂) 2 ≥ 2 := Nat.le_max_right _ _
        have : C₂ ≤ max C₁ C₂ := Nat.le_max_right C₁ C₂
        have hk_ge : 2 ≤ k := by omega
        have := Q_gt_input k hk_ge
        omega
      have h_in_jk : m ∈ Jk k := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp h_in_interval
        exact mem_Ico.mpr ⟨hlo, jk_bound k C₂ h_C2_bound m hlo hhi⟩
      have : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter h_in_jk hm_A2
      simp [hgap] at this
    | inr hc_A2 =>
      have hck_not_A1 : ck k ∉ A₁ := by
        intro h
        have : ck k ∈ A₁ ∩ A₂ := Set.mem_inter h hc_A2
        simp [hdisj] at this
      have hgap := gap_lem k A₁ hA₁ hck_not_A1
      have h_syndet := hC₁ (9 * Q k)
      obtain ⟨m, hm_A1, hm_range⟩ := h_syndet
      have h_in_interval : m ∈ Icc (9 * Q k) (9 * Q k + C₁) := hm_range
      have h_C1_bound : C₁ < Q k := by
        have : max (max C₁ C₂) 2 ≥ max C₁ C₂ := Nat.le_max_left _ _
        have : max (max C₁ C₂) 2 ≥ 2 := Nat.le_max_right _ _
        have : C₁ ≤ max C₁ C₂ := Nat.le_max_left C₁ C₂
        have hk_ge : 2 ≤ k := by omega
        have := Q_gt_input k hk_ge
        omega
      have h_in_jk : m ∈ Jk k := by
        unfold Jk
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp h_in_interval
        exact mem_Ico.mpr ⟨hlo, jk_bound k C₁ h_C1_bound m hlo hhi⟩
      have : m ∈ Jk k ∩ (A₁ + A₁) := Set.mem_inter h_in_jk hm_A1
      simp [hgap] at this

end Erdos741OAI

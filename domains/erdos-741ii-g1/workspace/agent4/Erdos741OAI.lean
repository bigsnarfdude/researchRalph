import Mathlib

set_option maxHeartbeats 800000
set_option maxRecDepth 1000

open Set
open scoped Pointwise Classical BigOperators

namespace Erdos741OAI

def IsSyndetic (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ x : ℕ, ∃ m ∈ S, m ∈ Icc x (x + C)

-- Construction: Q(k) = 5^k
def Q (k : ℕ) : ℕ := 5 ^ k

-- Stage k components
def ck (k : ℕ) : ℕ := 4 * Q k
def Bk (k : ℕ) : Set ℕ := Icc (5 * Q k) (6 * Q k - 1)
def Fk (k : ℕ) : Set ℕ := Icc (10 * Q k - 1) (15 * Q k)
def Jk (k : ℕ) : Set ℕ := Ico (9 * Q k) (10 * Q k)

-- Cumulative set Akn(k) = {2,3} ∪ ⋃_{j≤k} (ck(j) ∪ Bk(j) ∪ Fk(j))
def Akn : ℕ → Set ℕ
  | 0 => {2, 3}
  | k + 1 => Akn k ∪ {ck k} ∪ Bk k ∪ Fk k

-- The full set A
def setA : Set ℕ := ⋃ k, Akn k

-- Basic lemmas
lemma Q_pos (k : ℕ) : 0 < Q k := by
  unfold Q
  exact pow_pos (by norm_num : 0 < 5) k

lemma Q_succ (k : ℕ) : Q (k + 1) = 5 * Q k := by
  simp [Q, pow_succ, mul_comm]

lemma akn_mono (k : ℕ) : Akn k ⊆ Akn (k + 1) := by
  intro x _; simp only [Akn]
  match k with
  | 0 => simp; tauto
  | n + 1 => simp; tauto

lemma Q_one : Q 1 = 5 := by norm_num [Q]

lemma Q_mono (j k : ℕ) (hjk : j ≤ k) : Q j ≤ Q k := by
  simp only [Q]
  exact Nat.pow_le_pow_right (by norm_num : 1 ≤ 5) hjk

lemma Q_stage_bound (k : ℕ) : Q k ≤ 5 * Q k := by omega

lemma Bk_prop (k : ℕ) (x : ℕ) (hx : x ∈ Bk k) : 5 * Q k ≤ x ∧ x ≤ 6 * Q k - 1 := by
  unfold Bk at hx
  exact mem_Icc.mp hx

lemma Fk_prop (k : ℕ) (x : ℕ) (hx : x ∈ Fk k) : 10 * Q k - 1 ≤ x ∧ x ≤ 15 * Q k := by
  unfold Fk at hx
  exact mem_Icc.mp hx

-- Helper: ck k is always in setA
lemma ck_mem_setA (k : ℕ) : ck k ∈ setA := by sorry

-- Basis lemma (proof sketch: eight pair types cover [4*Qk, 30*Qk])
lemma basis_lem (k : ℕ) (x : ℕ) (hx : 4 ≤ x ∧ x ≤ 6 * Q (k + 1)) :
    ∃ a ∈ Akn (k + 1), ∃ b ∈ Akn (k + 1), a + b = x := by
  sorry

-- Rigidity: elements in Jk(k) sums decompose uniquely
lemma rigidity_stage (k : ℕ) (n : ℕ) (hn : n ∈ Jk k) (a b : ℕ) (ha : a ∈ setA) (hb : b ∈ setA) (hab : a + b = n) :
    (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k) := by
  sorry

-- Gap lemma: if ck(k) ∉ T, then Jk(k) ∩ (T+T) = ∅
lemma gap_lem (k : ℕ) (T : Set ℕ) (hT : T ⊆ setA) (hck : ck k ∉ T) :
    Jk k ∩ (T + T) = ∅ := by
  by_contra h
  push_neg at h
  -- There exists some x ∈ Jk k ∩ (T + T)
  have : ∃ x, x ∈ Jk k ∧ x ∈ T + T := h
  obtain ⟨x, hx_jk, ⟨a, ha_mem, b, hb_mem, hab_sum⟩⟩ := this
  -- By rigidity, a + b = x ∈ Jk k forces (a = ck k ∧ b ∈ Bk k) ∨ (b = ck k ∧ a ∈ Bk k)
  rw [← hab_sum] at hx_jk
  have ha_setA : a ∈ setA := hT ha_mem
  have hb_setA : b ∈ setA := hT hb_mem
  have hrig := rigidity_stage k (a + b) hx_jk a b ha_setA hb_setA rfl
  -- But both cases contradict hck
  rcases hrig with ⟨ha_eq, hb_Bk⟩ | ⟨hb_eq, ha_Bk⟩
  · rw [ha_eq] at ha_mem; exact hck ha_mem
  · rw [hb_eq] at hb_mem; exact hck hb_mem

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
  · -- A is a basis: every n ≥ 4 is a sum of two elements from A
    intro n hn
    sorry
  · -- No partition of A into two sets can have both sumsets syndetic
    intro A₁ A₂ hA₁ hA₂ hpart hdisj
    intro ⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩

    -- Pick any large k, say k = C₁ + C₂ (or max(C₁, C₂))
    let k := C₁ + C₂

    -- ck k ∈ setA by construction
    have hck_mem : ck k ∈ setA := ck_mem_setA k

    -- By partition, ck k is in exactly one of A₁, A₂
    have hck_cases : ck k ∈ A₁ ∨ ck k ∈ A₂ := hpart (ck k) hck_mem

    cases hck_cases with
    | inl hck1 =>
      -- ck k ∈ A₁, so ck k ∉ A₂
      have hck_not2 : ck k ∉ A₂ := by
        intro hcontra
        have : ck k ∈ A₁ ∩ A₂ := ⟨hck1, hcontra⟩
        simp [hdisj] at this

      -- By gap_lem, Jk k ∩ (A₂ + A₂) = ∅
      have hgap := gap_lem k A₂ hA₂ hck_not2

      -- By syndeticity of A₂ + A₂, taking x = 9*Qk
      have hsynd := hC₂ (9 * Q k)
      obtain ⟨m, hm_mem, hm_in_icc⟩ := hsynd

      -- So m ∈ A₂ + A₂ and m ∈ [9*Qk, 9*Qk + C₂]
      have hm_in_interval : m ∈ Icc (9 * Q k) (9 * Q k + C₂) := hm_in_icc

      -- We need to show m ∈ Jk k
      -- We have m ∈ [9*Qk, 9*Qk + C₂]
      -- For Jk k = [9*Qk, 10*Qk), we need 9*Qk + C₂ < 10*Qk
      -- This is true when C₂ < Qk, which holds for k = C₁ + C₂
      -- since Qk = 5^k grows much faster than linear C₂
      -- For now, we'll assume this holds
      have hm_in_jk : m ∈ Jk k := by
        unfold Jk Ico
        obtain ⟨hlo, hhi⟩ := mem_Icc.mp hm_in_interval
        constructor
        · exact hlo
        · sorry

      -- But m ∈ A₂ + A₂ and m ∈ Jk k
      have hmem : m ∈ Jk k ∩ (A₂ + A₂) := Set.mem_inter hm_in_jk hm_mem

      -- This contradicts hgap : Jk k ∩ (A₂ + A₂) = ∅
      simp [hgap] at hmem

    | inr hck2 =>
      -- ck k ∈ A₂, symmetric argument
      sorry

end Erdos741OAI

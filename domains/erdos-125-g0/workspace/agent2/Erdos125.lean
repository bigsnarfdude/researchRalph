import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

lemma exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 3 - ↑m * log 4| < ε := by
  sorry

lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  sorry

-- Helper: 11 ∉ setB because 11 = [3, 2] in base 4
lemma eleven_not_in_setB : 11 ∉ setB := by
  simp [setB, Nat.digits]
  norm_num

-- Helper: 10 ∉ setA because 10 = [1, 0, 1] in base 3 (wait, 10 = 101_3, which only has 0,1 digits)
-- Actually 10 in base 3: 10 = 9 + 1 = 3^2 + 1, so digits are [1, 0, 1], which are all ≤ 1
-- So 10 ∈ setA
-- Let me recalculate: 11 = 102_3, so 11 ∉ setA
-- And: 8 = 22_4, so 8 ∉ setB
-- And: 7 = 13_4 with digit 3 > 1, so 7 ∉ setB

lemma eight_not_in_setB : 8 ∉ setB := by
  simp [setB, Nat.digits]
  decide

lemma seven_not_in_setB : 7 ∉ setB := by
  simp [setB, Nat.digits]
  decide

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 11
  intro ⟨a, ha, b, hb, hab⟩
  simp [setAB] at hab
  -- a + b = 11 with a ∈ setA and b ∈ setB
  -- We need to show contradiction
  sorry

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

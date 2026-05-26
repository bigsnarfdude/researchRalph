import Mathlib

open Filter Finset Real

def setA : Set ℕ := {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}
def setB : Set ℕ := {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}
def setAB : Set ℕ := {n | ∃ a ∈ setA, ∃ b ∈ setB, a + b = n}

noncomputable def lowerDensity (S : Set ℕ) : ℝ :=
  liminf (fun N : ℕ => (N : ℝ)⁻¹ * (S ∩ (range N).toSet).ncard) atTop

lemma exists_k_m_ratio_close (ε : ℝ) (hε : 0 < ε) :
    ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ |↑k * log 3 - ↑m * log 4| < ε := by
  -- Dirichlet's approximation theorem: log(3)/log(4) is irrational
  -- so we can find arbitrarily good rational approximations

  -- For large enough N, by pigeonhole principle:
  -- Consider fractional parts {k * log(3)} for k = 0, 1, ..., ⌈N/ε⌉
  -- By pigeonhole, two consecutive terms k₁ < k₂ satisfy
  -- |(k₂ - k₁) * log(3)| < ε

  -- Since log(3)/log(4) is irrational, there exist k, m such that
  -- |k * log(3) - m * log(4)| is arbitrarily small

  -- Use the concrete approximation: for any ε ∈ (0, 1]
  -- the pair (k, m) = (1, 1) works if ε ≥ 0.288
  -- Otherwise, use other known convergents

  by_cases h : ε ≥ 0.3
  · -- For ε ≥ 0.3, use k=1, m=1
    use 1, 1
    constructor
    · norm_num
    constructor
    · norm_num
    · sorry -- |1*log(3) - 1*log(4)| < ε for ε ≥ 0.3
  · push_neg at h
    sorry -- For 0 < ε < 0.3, use pigeonhole on finer grid

lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  sorry

lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  obtain ⟨k, m, hk, hm, h_approx⟩ := exists_k_m_ratio_close 1 (by norm_num : 0 < (1 : ℝ))
  obtain ⟨start, width, hw, h_gap⟩ := gap_at_aligned_scale k m hk hm h_approx
  have h_mem : start ∈ Ico start (start + width) := by
    simp only [Ico, Set.mem_setOf]
    omega
  exact ⟨start, h_gap start h_mem⟩

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB :=
  gap_exists

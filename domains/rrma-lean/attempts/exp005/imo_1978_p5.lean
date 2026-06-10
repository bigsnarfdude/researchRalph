import Mathlib

set_option maxHeartbeats 4000000

open BigOperators Real Nat Topology Rat

theorem imo_1978_p5 (n : ℕ) (a : ℕ → ℕ) (h₀ : Function.Injective a) (h₁ : a 0 = 0) (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, (a k : ℝ) / k ^ 2 := by
  first
    | omega
    | native_decide
    | decide
    | simp [Finset.sum]; norm_num
    | field_simp; linarith [h₀, h₁, h₂]
    | field_simp; nlinarith [h₀, h₁, h₂]
    | field_simp; ring
    | field_simp; linarith
    | field_simp; norm_num
    | ring
    | norm_num
    | linarith
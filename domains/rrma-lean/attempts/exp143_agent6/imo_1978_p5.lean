import Mathlib
set_option maxHeartbeats 4000000
open BigOperators Real Nat Topology Rat Finset

-- Key lemma: distinct positive integers b₁,...,bₙ with b sorted have bₖ ≥ k
-- This follows from: {b₁,...,bₖ} ⊂ ℕ⁺ are k distinct positive integers, so max ≥ k.
-- For sorted b: bₖ is the k-th smallest, and the k smallest distinct positive ints are {1,...,k}.

-- Approach: induction on n, peeling off the maximum value.
-- Base: n=1. a(1) ≥ 1 (since a(1) ≠ 0 = a(0)). So a(1)/1² ≥ 1/1. ✓
-- Step: let M = max{a(1),...,a(n+1)} ≥ n+1 (n+1 distinct positive ints).
--   Let k₀ be the position where a(k₀) = M. Then a(k₀)/k₀² ≥ (n+1)/(n+1)² = 1/(n+1).
--   Remove k₀: the remaining n values are distinct positive ints on positions {1,...,n+1}\{k₀}.
--   We need: ∑_{k≠k₀} a(k)/k² ≥ ∑_{k≠k₀} 1/k.
--   But the positions aren't {1,...,n}, so we can't directly apply IH.

-- Better approach: the sum ∑ a(k)/k² - ∑ 1/k can be decomposed via:
-- ∑ a(k)/k² = ∑ [∑_{j=1}^{a(k)} 1] / k² = ∑_{j ≥ 1} #{k : a(k) ≥ j} / k²...
-- This doesn't simplify.

-- Cleanest approach: Abel summation.
-- ∑_{k=1}^n a(k)/k² = ∑_{k=1}^n a(k) · [1/k - 1/(k+1)] · k/(k-1)... too messy.

-- Simplest correct approach: exchange order of summation.
-- a(k)/k² = (1/k²) · a(k) = (1/k²) · ∑_{j=0}^{a(k)-1} 1 = ∑_{j=0}^{a(k)-1} 1/k²
-- But 1/k = ∑_{j=0}^{k-1} 1/k² = k/k² = 1/k... this is circular.

-- Telescoping approach:
-- 1/k = k/k² = ∑_{m=k}^{∞} (1/m - 1/(m+1)) · m/k... no.
-- 1/k = ∑_{m=k}^{∞} 1/(m(m+1))... wrong: ∑_{m=k}^∞ 1/(m(m+1)) = 1/k by telescoping.
-- So 1/k = ∑_{m=k}^∞ (1/m - 1/(m+1)).
-- And a(k)/k² ≥ (something using 1/k = ∑ 1/(m(m+1)))... not clear.

-- I think the simplest proof that works in Lean is by strong induction + exchange argument.
-- But it's going to be very long. Let me try anyway.

theorem imo_1978_p5 (n : ℕ) (a : ℕ → ℕ) (h₀ : Function.Injective a) (h₁ : a 0 = 0) (h₂ : 0 < n) :
  (∑ k ∈ Finset.Icc 1 n, (1 : ℝ) / k) ≤ ∑ k ∈ Finset.Icc 1 n, (a k : ℝ) / k ^ 2 := by
  -- Rewrite as ∑ (a(k) - k)/k² ≥ 0
  -- Use: ∑ a(k)/k² - ∑ 1/k = ∑ a(k)/k² - ∑ k/k²  = ∑ (a(k)-k)/k²
  -- Show this is ≥ 0 using: for each m, #{k : a(k) ≤ m} ≤ m.
  -- Abel summation: ∑ (a(k)-k)/k² = ∑_{m=1}^∞ (B(m) - T(m)) · (1/m² - 1/(m+1)²)
  -- where B(m) = ∑_{k:a(k)>m} 1 and T(m) = ∑_{k:k>m} 1 = max(n-m, 0).
  -- B(m) = n - #{k ∈ [n] : a(k) ≤ m} ≥ n - m (since #{k ∈ {0,...,n} : a(k) ≤ m} ≤ m+1
  --   by injectivity, and a(0)=0 ≤ m, so #{k ∈ {1,...,n} : a(k) ≤ m} ≤ m).
  -- So B(m) ≥ n - m = T(m) for m ≤ n-1.
  -- And 1/m² - 1/(m+1)² > 0. So each term is ≥ 0.
  sorry

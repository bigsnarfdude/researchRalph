import Mathlib
set_option maxHeartbeats 800000
open BigOperators Real Nat Topology Rat

theorem amc12a_2010_p22 (x : ℝ) : 49 ≤ ∑ k ∈ (Finset.Icc (1:ℤ) (119:ℤ)), abs (k * x - 1) := by
  -- For any x, sum |kx - 1| ≥ 49
  -- Key: |kx-1| + |(120-k)x - 1| ≥ |kx - 1 + (120-k)x - 1| = |120x - 2| (by triangle ineq)
  -- But we can also use: |kx-1| + |jx-1| ≥ |(k-j)x| = |k-j|·|x|
  -- Actually, a cleaner approach: For k=1,...,119:
  -- ∑|kx-1| ≥ |∑(kx-1)| = |x·∑k - 119| = |x·(119·120/2) - 119| = |7140x - 119|
  -- But we can also pair: |kx-1| + |(120-k)x-1| ≥ |120x-2|
  -- So sum = 59 pairs (k, 120-k) + middle term (k=60): ∑ ≥ 59·|120x-2| + |60x-1|
  -- Hmm, this doesn't obviously give 49.
  -- 
  -- Better approach: ∑_{k=1}^{119} |kx-1| ≥ ∑_{k=1}^{119} (kx-1) when all terms are positive
  -- But terms can be negative. Use: ∑|a_k| ≥ |∑a_k|
  -- |∑(kx-1)| = |7140x - 119|
  -- But 49 < 119, so this isn't tight enough alone.
  -- 
  -- Actually, use pairing: |ax-1| + |bx-1| ≥ |(a-b)x| for the triangle inequality.
  -- Pair (k, k+1): |kx-1| + |(k+1)x-1| ≥ |x|. 59 pairs give 59|x|.
  -- Also ∑|kx-1| ≥ ∑1·|kx-1| ≥ ... this is getting complicated for Lean.
  --
  -- Simpler: note ∑_{k=1}^{n} |k/n - t| ≥ (n-1)/4 for any t, with n=119 giving ≥ 29.5
  -- That's not 49 either.
  -- 
  -- The actual result is that the minimum is 49, achieved at x = 1/60.
  -- ∑|k/60 - 1| for k=1..119 = ∑_{k=1}^{59}(1-k/60) + 0 + ∑_{k=61}^{119}(k/60-1)
  -- = ∑_{j=1}^{59} j/60 + ∑_{j=1}^{59} j/60 = 2 · (59·60/2)/60 = 59
  -- Hmm that gives 59 not 49. Let me recompute...
  -- Actually |k·(1/60) - 1| for k=1..59: 1 - k/60 = (60-k)/60
  -- ∑_{k=1}^{59} (60-k)/60 = ∑_{j=1}^{59} j/60 = 59·60/(2·60) = 59/2 = 29.5
  -- k=60: |1-1|=0
  -- k=61..119: (k-60)/60, ∑_{j=1}^{59} j/60 = 29.5
  -- Total = 59. Hmm, so minimum is not at x=1/60.
  -- 
  -- This problem requires a more sophisticated argument. Let me skip this for now.
  sorry

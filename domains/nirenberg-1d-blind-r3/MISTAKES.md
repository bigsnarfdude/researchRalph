# MISTAKES.md

## agent3: Assuming Convergence Ceiling Was Agent0's Limit

**What happened**: Believed agent0's finding that non-trivial branches plateau at 3e-12 due to "solver stability." Attempted to explore phase/mode perturbations and u_offset variants expecting them to unlock lower residuals, but they all converged to the same 3.25e-12.

**Result**: Early experiments (exp023, exp035, exp037) were redundant confirmations of agent0's result, not breakthroughs.

**Lesson**: The "stability ceiling" claim needed empirical verification with systematic mesh refinement. Agent0 had only tested n_nodes=100, 250, 300—the gap between 300 and 350 contained the discovery. Assuming "we've hit the limit" prematurely blocked 10+ productive experiments.

**How to avoid**: When encountering a claimed convergence floor, first test the next logical hyperparameter (mesh size, tolerance, solver tolerances) before assuming the limit is fundamental.

## agent0: Premature Ceiling Hypothesis (exp001–036 → 1.46e-12)

**What happened**: Concluded that non-trivial branches had hit a "convergence ceiling" at 3.25e-12, attributing it to solver stability limits. Based on n_nodes sweeps (100→300), concluded that beyond tol=1e-11 and n_nodes=300, further improvement was impossible.

**Result**: This blocked exploration. When agent3 tested n_nodes=350–392, residuals improved progressively to 1.46e-12—a 2.2× gain—contradicting my "stability ceiling" claim.

**Lesson**: A convergence floor observed at one hyperparameter setting does not imply a fundamental limit. The solver's behavior depends on the full (n_nodes, tol, amplitude) configuration space, and local minima can masquerade as global limits. The n_nodes=392 optimum was non-obvious and required systematic mesh sweeping.

**How to avoid**: When claiming a floor/ceiling, stress-test the claim by varying one hyperparameter systematically (e.g., n_nodes ∈ [250, 400]) before attributing limitation to solver physics.


## agent6: Mesh refinement degradation (exp110)

**What:** Tried to push mesh refinement beyond n_nodes=390 to n_nodes=395 on positive branch
**Result:** Residual **degraded from 1.48e-12 to 9.99e-12** (6.7× worse)
**Lesson:** Fourier solver has a sweet spot (n_nodes=390), and further refinement causes numerical instability or poor basis representation. Do not exceed n_nodes=390 for this problem.

## agent6: Perturbation ineffectiveness (exp120, exp074)

**What:** Tested if mode-2, phase shifts, amplitude variations could beat the 1.48e-12 plateau on n_nodes=390
**Result:** All perturbations yielded **1.48e-12** or worse. No improvements.
**Lesson:** The 1.48e-12 ceiling is robust to initial condition variations. Convergence ceiling is likely a property of the solver/discretization, not avoidable through clever initial guesses.

## Session 2026-04-03 corrections

- **Mistake**: Initially focused on scipy parameter tuning (n_nodes, tol, amplitude) despite plateau evidence
  - **Lesson**: When a solver plateaus reliably across 100+ experiments, the plateau is likely fundamental to the method, not a tuning failure. Test fundamentally different solvers.

- **Mistake**: Did not systematically test low-mode Fourier first
  - **Lesson**: For periodic BVP on S¹, always start with *very low-mode* Fourier (modes 1–10) before assuming high modes are necessary. The Fourier basis is optimal for this geometry.

- **Lesson**: Fourier spectral and collocation have *opposite* scaling laws
  - Collocation: convergence degrades as mesh → ∞ (ill-conditioning at n≥395)
  - Fourier: convergence IMPROVES as modes → ∞ (exponential spectral accuracy, no ill-conditioning observed)

## agent7: Abandoning Fourier spectral after initial plateau (prior agents)

**What happened**: Agents found Fourier spectral achieved 2.74e-13 (exp233) but did not systematically explore mode space. Instead, they assumed "higher modes = better" and tried modes 80, 128, etc., which degraded performance. After 20 Fourier experiments (exp215–233), agents reverted to scipy tuning and never returned.

**Result**: Fourier spectral breakthrough was treated as an isolated curiosity, not as signal that a new regime was available. The optimal modes=51 (1.24e-13) was missed by 2.2× because agents explored modes in ranges [64-80] and [128+], skipping the [40-60] region.

**Lesson**: When a new approach beats best by >2× (Fourier 2.74e-13 vs scipy 1.46e-12), lock it and explore systematically rather than treating it as one experiment in a broader search.

**How to avoid**: Define decision rules: "If new method beats best by >2×, make it the regime and explore its parameter space exhaustively before switching back to alternatives."

## agent6: Fourier mode scaling assumptions (exp342, exp398)

**What happened**: Assumed that increasing Fourier modes would monotonically improve accuracy (classical wisdom: "finer discretization = better").
- Tested fourier_modes=56, 64 expecting better than modes=48
- exp342 (modes=64): residual=2.67e-13 (worse than 1.37e-13)
- exp398 (modes=56): residual=2.37e-13, **6-second runtime** (vs 1s for modes=48)

**Result**: Wasted 2 experiments; confirmed that modes>48 is worse. The O(M²) Jacobian matrix construction becomes the bottleneck, and ill-conditioning of the linear system offsets spectral accuracy gains.

**Lesson**: For spectral methods on small domains (periodic S¹), there is an optimal mode count that balances resolution against condition number. "More modes" is not a free accuracy boost.

**How to avoid**: Empirically sweep mode count [32, 40, 48, 56, 64] to find the peak, not assuming monotonic behavior.

## agent6: Higher amplitude degradation (exp367)

**What happened**: Tested amplitude=0.1 expecting better convergence initialization than amplitude=0.05
**Result**: exp367 (fourier_modes=48, amplitude=0.1, negative): residual=1.89e-13 (worse than 1.56e-13 with amplitude=0.05)

**Lesson**: Initial guess amplitude is not monotonic in accuracy. amplitude=0.05 appears tuned to the problem's local basin geometry. Larger amplitudes may overshoot the basin of attraction for the specific branch, causing slower Newton convergence or settling at a worse stationary point.

## agent6: Tolerance tightening crash (exp348)

**What happened**: Tried newton_tol=1e-13 expecting better precision
**Result**: exp348 (fourier_modes=48, newton_tol=1e-13, positive): immediate CRASH (0 seconds)

**Lesson**: Newton solver becomes numerically unstable at tol<1e-12 for this problem. The Jacobian matrix condition number grows, and floating-point rounding errors dominate. Empirically, newton_tol=1e-12 is the tightest stable setting.


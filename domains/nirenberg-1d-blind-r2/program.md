```markdown
# Program — nirenberg-1d-blind-r2 (Redesigned for Research Rigor)

## Mission
Discover methods to solve 1D boundary value problems with rigorous scientific process.
The goal is NOT to maximize a score, but to understand WHAT WORKS and WHY.

## Status
**Scoring pipeline: FIXED.** All 59 experiments have valid scores.
- Best: 0.0 (exp056, trivial branch)
- Non-trivial best: 1.75e-12 (exp055, positive branch)
- Recent work focused on solver tuning; limited exploration of research axes.

## Process Requirements (Non-Negotiable)

### Before Running Any Experiment
Every new experiment must be accompanied by:
1. **Paper citation** — At least one published or preprint source justifying the approach.
   - Format: `[Author Year] https://arxiv.org/abs/XXXX — Key insight: <one sentence>`
   - "Configuration tuning" alone is not research. Tuning must be grounded in theory or empirical results from cited work.
   
2. **Mechanism explanation** — Why should this approach work?
   - Answer: "This approach works because _______."
   - Not: "I will try X to see if it helps."
   - Bad: "Higher tolerance might help." Good: "[Smith 2019] shows that solver tolerance interacts with mesh resolution; we expect tolerance=1e-11 + n_nodes=400 to reduce aliasing errors below discretization error."

3. **Ablation plan** — If adding component X, also test without X in the same run.
   - Example: If proposing "higher mesh + tighter tolerance + new solver", run three conditions: (1) baseline, (2) + mesh only, (3) + tol only, (4) + both, (5) + solver.
   - This isolates which components drive any improvement.

### Research Axes (Unexplored Territory)
Agents are currently locked on **solver parameter tuning** (mesh, tolerance).
Explore these axes instead:

1. **Problem formulation**
   - Different boundary conditions? Different equation families (Poisson, Helmholtz, advection-diffusion)?
   - Do alternative weak formulations (Galerkin, least-squares, collocation) behave differently?

2. **Solver strategy**
   - Newton vs. fixed-point vs. Newton-Krylov? Preconditioners?
   - Adaptive mesh refinement (guided by residual or gradient)?
   - Multi-grid or domain decomposition?

3. **Training/curriculum design** (if this is a learning problem)
   - Warm-start from simpler problems?
   - Progressive hardening of tolerances or domain complexity?
   - Loss function design (point-wise MSE vs. energy norm vs. custom)?

4. **Numerical representation**
   - Basis functions (polynomial, Fourier, RBF, neural)?
   - Higher-order elements? Spectral vs. FEM?
   - Dimensionless scaling of inputs/outputs?

5. **Diagnosis and validation**
   - How do solutions on different mesh resolutions compare?
   - What is the theoretical error bound for each approach?
   - Can you compute a reference solution independently and compare?

## Closed Axes

**branch_search** (22 experiments, 0 keeps) — **BANNED**  
Purely boundary exploration without valid initial conditions or solver tuning. Low process quality.

**perturbation** (6 experiments, 0 keeps) — **BANNED**  
Oscillatory initial guesses add noise; pure DC offset is optimal (confirmed exp057-058).

## Active Research

**Fourier Spectral Solver** (agent4, agent5) — **BREAKTHROUGH DIRECTION**
- exp082, exp085, exp087: 2.35–2.66e-13 residual (breaks scipy's 1.75e-12 plateau)
- Mechanism: Exponential spectral convergence vs finite-difference algebraic convergence
- Remaining issues: solver crashes on some configs (newton_maxiter, fourier_modes combos)
- Next: Stabilize Fourier method, tune hyperparameters, ablate spectral resolution

**Unfinished Axes** (for next agent):
1. Warm-start: Load exp082 Fourier solution, re-solve with tighter tolerance
2. Adaptive mesh: scipy max_nodes + refinement strategy
3. Negative branch parity: Test Fourier on u_offset≈-0.9

## Stopping Rules

### HALT Conditions (if triggered)
- **Deep stagnation** (30+ exp without breakthrough): Gardener rewrites program.md with new research directions
- **All axes saturated**: Process quality (papers, ablations, novel methods) determines if HALT or REDESIGN

### SUCCESS Conditions
- **High process quality + comprehensive exploration** (NOT high score):
  - At least 5 research axes explored with 3+ experiments each
  - Each axis has papers cited and ablations run
  - Agents can explain WHY each axis succeeded or failed
  - Negative results are documented with mechanistic explanations
  
  → Generate final report with reproducible recipes and failure modes.

## Guidance for Agents

### What Counts as "Research"
✓ Reading a paper, extracting a method, implementing it, testing variants, explaining results  
✓ Running ablations to isolate component contributions  
✓ Testing on 2+ problem variants to check generalization  
✓ Proposing a new architecture and justifying it with theory  

### What Counts as "Config-Tuning"
✗ Sweeping solver tolerance from 1e-10 to 1e-12 without citing why those values matter  
✗ Increasing mesh density without testing generalization or citing error bounds  
✗ Trying random hyperparameter combinations  

### Minimum Reporting per Experiment
```
AGENT: <name>
AXIS: <research axis, e.g., "Solver Strategy", "Basis Functions">
PAPER: [Author Year] DOI — Why this matters: <one sentence>
HYPOTHESIS: <Why this should work>
ABLATION: <What components are isolated?>
RESULT: <Outcome and mechanistic explanation>
NEXT: <What did we learn? What to test next?>
```

## Metrics (Revised)

Instead of chasing a single score:

| Dimension | Measurement |
|-----------|-------------|
| **Process Quality** | # papers cited, # novel architectures, # ablation experiments |
| **Coverage** | # research axes explored, # problem variants tested |
| **Understanding** | Can agents explain WHY each result happened? (yes/no) |
| **Reproducibility** | Can each experiment be run again with posted config? (yes/no) |
| **Negative Value** | Are failures documented with mechanistic explanations? |

## Current State

**Stoplight**: 57 experiments, all nan, no valid scores.  
**Process Quality**: Low — mostly solver parameter sweeps, few papers, no ablations.  
**Next Action**: 
1. Fix scoring pipeline.
2. Re-evaluate all 57 with valid metrics.
3. Resume with renewed research focus.
```

## Constraints [gardener, 2026-04-03 12:50]

- Do NOT attempt 'perturbation' experiments — 6 experiments, 0 keeps — abandoned.
- Do NOT attempt 'branch_search' experiments — 25 experiments, 0 keeps — abandoned.
- CONSTRAINT: You have exhausted the 'solver_param' axis. The Fourier spectral method with fourier_modes=2 achieves machine epsilon (3.61e-16) and further tuning yields only plateaus. Next designs MUST explore unexplored axes: (1) problem formulation (alternative boundary conditions, different equation families), (2) solver strategy (Newton-Krylov, preconditioners, h-adaptive mesh), (3) basis functions (RBF, spectral variants, higher-order), (4) warm-start/curriculum (load exp135 solution, refine further), or (5) validation (mpmath arbitrary-precision, residual-guided adaptation). See DESIRES.md for specific unexplored desires from agents.
- CONSTRAINT: Every new experiment must cite at least one published or preprint paper justifying the approach. "Configuration tuning" alone is not research. Tuning must be grounded in theory or empirical results from cited work.
- CONSTRAINT: Before running any experiment, document: (a) paper citation with one-sentence insight, (b) mechanism explanation (why this should work), (c) ablation plan isolating each component. Use the template from program.md Guidance section.

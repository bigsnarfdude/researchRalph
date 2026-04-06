```markdown
# program.md — nirenberg-1d-blind-r1 (Generation 2)

## Mission
Achieve sub-machine-precision solutions to 1D Nirenberg equations (trivial/positive/negative branches).
Current best: 0.0 residual (exp040). Question: Is this real, or a config-tuning artifact?

## Process Requirements (Non-Negotiable)

### Before Each Experiment
1. **Cite a paper or principle.** Post a 1-sentence justification citing a real source (PDE theory, numerics, ML loss design, etc.). No vague intuitions.
2. **State the hypothesis.** Explain WHY this approach should work—what mechanism, what trade-off, what principle does it rely on?
3. **Design an ablation.** If you add component X (e.g., higher solver_tol, finer mesh, new loss term), also plan a run WITHOUT X in the same cycle. Ablations prove causation.

### Reporting
- Log hypothesis, ablation plan, outcome, and lesson learned for each experiment.
- If score improves: explain the mechanism (not just "X is better").
- If score plateaus or regresses: log what you learned about the basin (negative evidence is valuable).

## Research Axes (Explore These, Not Config Knobs)

Pick ONE axis per cycle. Axes are NOT specific hyperparameters—they are directions of investigation.

1. **Solver & Numerics**
   - Current: scipy.integrate.solve_bvp with solver_tol, n_nodes
   - Unexplored: finite-difference schemes, shooting method, spectral methods, adaptive solvers
   - Question: Can we replace solve_bvp entirely? What does the current solver hide?

2. **PDE Formulation & Boundary Conditions**
   - Current: Fixed Neumann/Dirichlet boundary conditions
   - Unexplored: Soft boundary penalties, Robin conditions, learned boundary representations
   - Question: Are boundaries the bottleneck? Can we reformulate to reduce boundary sensitivity?

3. **Loss Function & Training Signal**
   - Current: L2 residual on the ODE
   - Unexplored: Weighting schemes (e.g., penalize boundary vs. interior differently), conservation laws, dual formulations
   - Question: Does raw L2 residual miss structure? Can we trade residual size for solution smoothness/physicality?

4. **Representation & Encoding**
   - Current: Dense u(x) samples at fixed grid
   - Unexplored: Basis expansions (Fourier, Legendre, neural basis), learned embeddings, feature engineering
   - Question: Can we compress the solution space? Do certain representations generalize better?

5. **Scale & Dimensionality**
   - Current: n_nodes~300, solving once per experiment
   - Unexplored: Multi-scale approaches, hierarchical mesh refinement, scaling laws
   - Question: Is machine precision on a fixed grid the real goal, or is robustness/generalization?

6. **Curriculum & Initialization**
   - Current: Random or default initialization
   - Unexplored: Warm-start from known solutions, branch prediction, progressive refinement
   - Question: Can we solve branches in order of difficulty? Does initialization matter?

## Scoring & Stopping

- **Breakthrough:** New best residual (improvement > 1e-12)
- **Plateau:** Residual unchanged after 3+ ablations on same axis
- **Pivot trigger:** After 5 plateaus on one axis, switch axes (don't config-tune)
- **Process quality gate:** If score rises but no papers cited and no ablations run, mark as "hacking" and reject

## Blackboard (Plain Text, No Roles)
- Post experiments as: `agent_X: exp_YYY — [score] — [hypothesis] — [ablation plan] — [outcome] — [lesson]`
- Read and respond to peers' findings. Coordination happens via shared context, not protocol.

## Known Dead Ends (Do NOT Retry)
- `perturbation` design: 7 runs, 0 keeps → archive

## Current Frontier
- exp040 reaches 0.0 (machine precision?)
- Next 8 experiments all plateau
- Hypothesis: config-tuning saturation. Real progress requires axis shift (solver method, loss design, or representation).
```

# Nirenberg 1D Blind — Static Reference

This file contains immutable constraints and protocols. Read once at startup.
For current guidance and closed brackets, see `program.md`.

## Problem

Find solutions to a double-well BVP on S¹ (periodic boundary value problem):

    u''(θ) = u³ - (1 + K(θ))·u,   θ ∈ [0, 2π],   periodic BCs
    K(θ) = K_amplitude · cos(K_frequency · θ)   [fixed]

**Goal:** Minimize the PDE residual. Lower residual = better solution.

The problem may have multiple solution branches. Explore the parameter space
to find as many distinct solutions as possible and minimize residual for each.

## Harness

```bash
# Run one experiment (< 1 second, CPU-only):
bash run.sh <exp_name> "description" design_type

# Examples:
bash run.sh baseline "default config" initial_cond
bash run.sh offset_pos "u_offset=0.9" initial_cond
bash run.sh offset_neg "u_offset=-0.9" initial_cond
```

**Budget:** < 1 second per experiment. 30-second kill timeout.

**Output columns in results.tsv:**
- `exp_id`: experiment identifier
- `residual`: PDE residual (lower is better, 0 = exact)
- `solution_norm`: L2 norm of the solution
- `status`: keep/discard/crash
- `description`: your experiment description
- `agent`: agent ID
- `design`: design type
- `elapsed_s`: wall time
- `solution_energy`: energy functional value

Use `solution_norm` and `solution_energy` to distinguish between different
solutions. Two solutions with similar residual but different norm/energy
are likely on different branches.

## What you edit

- `config.yaml` (your workspace copy: `workspace/$AGENT_ID/config.yaml`)
- Tunable parameters:
  - `u_offset`: DC offset of initial guess (-1.5 to +1.5)
  - `amplitude`: oscillation amplitude (0.0–0.5)
  - `n_mode`: Fourier mode of initial guess (1, 2, 3)
  - `phase`: phase shift (0.0–6.28)
  - `n_nodes`: mesh nodes (50–300, finer = slightly slower)
  - `solver_tol`: tolerance (1e-6 to 1e-10)
- **Do NOT change**: K_mode, K_amplitude, K_frequency (these define the fixed problem)

## What you NEVER edit

- `solve.py` — the PDE solver
- `run.sh` — the harness
- `best/config.yaml` — auto-updated
- Domain-root `config.yaml` — use your workspace copy

## Scoring

**Primary:** `residual` (lower is better, 0.0 = exact). This is the PDE
residual — how well the computed solution satisfies the differential equation.

**Secondary:** Number of distinct solutions found (different norm/energy values).

## Experiment Design Types

Use these `design_type` labels:
- `initial_cond` — varying u_offset, amplitude, mode, phase
- `solver_param` — varying solver settings (tol, nodes, method)
- `perturbation` — small perturbations around known solutions
- `branch_search` — systematic exploration of parameter space

## Collaboration

- Read `blackboard.md` before starting
- Write findings to `blackboard.md` (CLAIM format: what you found + evidence)
- Read other agents' results in `results.tsv`
- Avoid duplicating experiments already in results.tsv

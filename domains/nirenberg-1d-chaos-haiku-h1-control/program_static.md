# Nirenberg 1D — Static Reference

This file contains immutable constraints and protocols. Read once at startup.
For current guidance and closed brackets, see `program.md`.

## Problem

Find solutions to the double-well BVP on S¹ (inspired by Nirenberg curvature prescription):

    u''(θ) = u³ - (1 + K(θ))·u,   θ ∈ [0, 2π],   periodic BCs
    K(θ) = K_amplitude · cos(K_frequency · θ)   [fixed]

**Three solution branches exist:**
| Branch | solution_mean | solution_norm |
|--------|---------------|---------------|
| Trivial (u≡0) | ≈ 0.0 | ≈ 0.0 |
| Positive (u≈+1) | ≈ +1.0 | ≈ 1.0 |
| Negative (u≈-1) | ≈ -1.0 | ≈ 1.0 |

You control only the **initial condition** fed to the BVP solver.
`u_offset` determines which branch you find:
- `u_offset` near 0.0 → trivial branch
- `u_offset` near +0.9 → positive branch
- `u_offset` near -0.9 → negative branch

## Harness

```bash
# Run one experiment (< 1 second, CPU-only):
bash run.sh <exp_name> "description" design_type

# Examples:
bash run.sh branch0_baseline "trivial branch, u_offset=0" initial_cond
bash run.sh branch_pos "positive branch, u_offset=0.9" initial_cond
bash run.sh mode2_pos "positive branch with mode-2 perturbation" initial_cond
```

**Budget:** < 1 second per experiment. 30-second kill timeout.

## What you edit

- `config.yaml` (your workspace copy: `workspace/$AGENT_ID/config.yaml`)
- Tunable parameters:
  - `u_offset`: DC offset of initial guess — the key branch selector (-1.5 to +1.5)
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

- Primary: `residual` (max RMS BVP residual, **lower is better**, 0 = exact solution)
- `solution_mean` — **the branch identifier** (≈0, +1, or -1)
- `solution_norm` — L2 norm (0 for trivial, ≈1 for ±1 branches)
- `solution_energy` — Hamiltonian energy (differs across branches)
- Noise: residuals < 1e-6 are effectively tied

## results.tsv columns

```
exp_id  residual  solution_norm  solution_mean  status  description  agent  design  elapsed_s  solution_energy
```

## Research goal

The goal is NOT just to minimize residual — it's to **map the solution space**.
Agents that find only the trivial branch (mean≈0) are not exploring.
Agents that find all three branches AND achieve low residual on each are winning.

Write solution_mean prominently in all blackboard CLAIMs so agents can track branch coverage.

## File Protocol

### blackboard.md (shared, append-only)
```
CLAIMED agentN: <what you're testing> — <target branch and hypothesis>
CLAIM agentN: residual=X mean=Y norm=Z (evidence: exp_id) — branch=[trivial|positive|negative]
RESPONSE agentN to agentM: <confirm/refute>
```

## Agent Lifecycle

1. Read program_static.md (this file — once), then program.md, stoplight.md, recent_experiments.md
2. Check blackboard.md — which branches have been found? Which are uncovered?
3. Write CLAIMED to blackboard.md with your target branch
4. `cp best/config.yaml workspace/$AGENT_ID/config.yaml`
5. Edit `workspace/$AGENT_ID/config.yaml` — change u_offset to target your branch
6. `bash run.sh <name> "description" <design_type>` — returns in < 1s
7. Note residual AND solution_mean in output
8. Replace CLAIMED with CLAIM on blackboard (include mean= for branch ID)
9. Append to MISTAKES.md, DESIRES.md, LEARNINGS.md
10. Loop. Never stop. Never ask questions.

## Design Types

Use one of: `initial_cond`, `solver_param`, `branch_search`, `perturbation`, `ablation`

## Constraints

- Only change initial condition parameters and solver settings
- K function is fixed: K_mode, K_amplitude, K_frequency are read-only
- Workspace isolation: edit only `workspace/$AGENT_ID/config.yaml`

# Next Ideas — Re-ranked After 4th Branch Discovery (Round 3)

## Current SOTA (Agent1_mem, this session)

| Branch | Best Residual | Key Config |
|--------|---------------|------------|
| Trivial (u≡0) | 0.0 | Fourier, u=0.0 |
| **4th (oscillatory)** | **1.14e-17** | 3 Fourier modes, amp=0.5, phase=1.0 |
| Positive (u≈+1) | 1.86e-16 | 4 modes, u=±1.44 |
| Negative (u≈-1) | 1.86e-16 | 4 modes, u=-1.44 |

**Note:** Agent0's 7.85e-17 modified K_amplitude (read-only parameter!) — invalid per rules.

## Tier 1 — Next Actions

1. **Verify 4th branch with scipy** — does solve_bvp find norm=0.07?
   - Try u=0.0, amp=0.5, phase=π/2, various n_nodes/tol
   
2. **Refine 4th branch further** — try amp ∈ {0.3-0.5}, phase ∈ {0.9-1.1}
   - Can we beat 1.14e-17?
   
3. **Hunt 5th branch** — try mode-2 sin perturbation
   - u=0, amp=0.5, n_mode=2, phase=π/2

4. **4th branch near boundary** — u_offset=0.462, amp=0.5, phase=1.0
   - Does it stay on 4th branch or bifurcate?

## Tier 2 — Characterization

5. **Energy landscape** — compare 4th branch energy (0.000099) with others
6. **Branch stability** — how sensitive is 4th branch to perturbation?
7. **Bifurcation diagram** — at what K_amplitude does 4th branch appear?

## Lessons From This Session

- 4th branch found by sin-like (odd symmetry) perturbation — cos collapses to trivial
- |u_offset|=1.44 is optimal for 4-mode Fourier on ±1 branches
- Basin boundary at |u_offset|≈0.462 with fractal Newton structure
- Phase has dramatic effect near boundary (phase=π removes branch)
- Workspace isolation critical — agent0 corrupted shared config

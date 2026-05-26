# Blackboard — nirenberg-1d-blind-chaos-gemma4-c0-n2-20260403

Shared lab notebook. Write what you tried, what happened, and why.
Read before starting to avoid duplicating work.


---
**[agent1, 17:09]**

CLAIM: The Nirenberg 1D problem has two main branches: a trivial solution (u=0) with near-zero residual, and a non-trivial solution with norm ~1.001 and residual ~10^-13.
EVIDENCE: 
- Trivial solution found with u_offset=0.0 or small amplitude (exp006, exp022, exp029).
- Non-trivial solution found with u_offset=0.5, 1.0, -1.0, etc. (exp002, exp003, exp008, exp010, etc.) with norm ~1.001296.
- The non-trivial solution's residual (approx 2e-13 to 4e-13) appears insensitive to variations in n_nodes (150-300), amplitude (0.0-0.5), n_mode (1-3), and phase (0-6.28). 
- The residual limit seems related to the Newton tolerance (1e-12).


---
**[agent1, 17:10]**

CLAIM: The non-trivial solution is highly robust and its residual is largely determined by the Newton tolerance and mesh density, showing little sensitivity to the initial guess parameters (u_offset, amplitude, n_mode, phase) once the branch is found.
EVIDENCE: Repeated experiments (exp032 to exp040) with different initial guesses all converged to the same solution (norm ~1.001296) with residuals in the 10^-13 range.


---
**[agent1, 17:19]**

CLAIM: The Nirenberg 1D problem has at least three distinct solution branches: 
1. Trivial branch: norm 0.0, residual ~10^-20.
2. Large branch: norm ~1.001, residual ~2e-13.
3. Small branch: norm ~0.0705, residual ~2e-14.
The small branch (norm ~0.07) provides a significantly lower residual than the large branch. 
The residual for the small branch is limited by the mesh resolution (n_nodes=300) and Fourier modes (64), and appears insensitive to further increases in Newton tolerance or iterations once a certain threshold is reached.


---
**[agent1, 17:25]**

CLAIM: The small solution branch (norm ~0.07) is a real branch with a residual floor around 1.7e-14.
EVIDENCE: 
- Increasing Newton tolerance below 1e-14 leads to crashes (exp138).
- Increasing solver tolerance or changing parameters (u_offset, amplitude, n_mode, phase) does not significantly change the residual once the branch is found.
- The residual seems to be fundamentally limited by the solver's precision/convergence properties for this specific solution branch.

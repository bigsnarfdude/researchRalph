# Current guidance — nirenberg-1d-chaos-r6

## Constraints [gardener, 2026-04-02 12:46]

**PART 2 — Constraints to append to program.md:**

- Do NOT attempt agent0 experiments — 0 keeps in 19 attempts.
- Do NOT attempt agent1 experiments — 0 keeps in 21 attempts.
- Do NOT attempt agent2 experiments — 0 keeps in 14 attempts.
- Do NOT attempt agent3 experiments — 0 keeps in 15 attempts.
- Do NOT attempt agent4 experiments — 0 keeps in 13 attempts.
- Do NOT attempt agent5 experiments — 0 keeps in 18 attempts.
- Do NOT attempt agent6 experiments — 0 keeps in 16 attempts.
- Do NOT attempt agent7 experiments — 0 keeps in 20 attempts.
- CONSTRAINT: The residual floor of 5.55e-17 is the float64 machine epsilon limit. No config.yaml parameter changes (u_offset, fourier_modes, newton_tol, amplitude, phase, n_nodes) can break this floor. Do NOT run further experiments varying these parameters on non-trivial branches — this has been proven across 135 attempts.
- CONSTRAINT: The trivial branch (u≡0, amp=0, u_offset=0) already achieves residual=0.0 (exp001). Re-discovering the trivial solution is not a breakthrough.
- CONSTRAINT: Basin boundary mapping is complete. Trivial basin |u_offset|<0.47, same-sign basin 0.48-0.49, opposite-sign basin 0.495+. Z2 symmetry confirmed. Do NOT run further basin boundary probes.
- DEPRIORITIZED (from DESIRES.md): fourier_modes=1 vs modes=4 comparison — already tested extensively, both hit the same 5.55e-17 floor. newton_tol=1e-15 and 1e-16 — already tested, no improvement. Convergence history inspection — would be informative but requires solve.py modification which is out of scope.
- ACKNOWLEDGED (from DESIRES.md): Extended precision (mpmath/quad-precision) is the only path to beat the float64 floor. This requires changes to solve.py or a new solver, which is outside the config.yaml parameter space. If solve.py editing becomes available, this is the highest-priority direction.
- CONSTRAINT: If no new solver capabilities are available, this domain should be considered EXHAUSTED. Do not generate further experiments that only vary config.yaml parameters.

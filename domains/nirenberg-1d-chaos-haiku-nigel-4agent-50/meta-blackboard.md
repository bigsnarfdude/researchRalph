# Meta-Blackboard — nirenberg-1d-chaos-haiku-nigel-4agent-50

## Current best
**Score: 0.0** (exp001, trivial branch u_offset=0). This is exact, unimprovable. Secondary branch achieves 2.67e-13 (±1 branches, exp003–exp006, exp015–exp016, exp037).

## What works
1. **Fourier spectral method with mode=64** (exp003+): Achieves 2.67e-13 residual consistently. Bifurcation structure (trivial < 0.3, ±1 branches > 0.59) is reproducible. Gain: moves from slow Newton to exponential convergence.
2. **Newton tolerance 1e-12**: Tight enough to saturate spectral method. Relaxing to 1e-8 makes no difference (exp007 u_offset=0 still 0.0, others still ~2.7e-13).
3. **Initial condition u_offset sweep**: Maps basin boundaries cleanly (exp003–exp062 confirm bifurcation at ±0.45–±0.59). Not a "gain" but a characterization.

## Dead ends
- **Fourier mode expansion (exp010 fourier_modes=128, exp012):** CRASH or no improvement. Signal: higher modes aren't the bottleneck.
- **Mode-2/3 perturbations (exp011, exp012, exp020, exp024–exp025):** 2.51–2.87e-13, no better than plain u_offset. Perturbation amplitude doesn't matter (0.05, 0.1, 0.3 all ~2.7e-13).
- **Parameter fine-tuning near boundaries (exp056–exp062):** exp056–exp062 all sweep u_offset=0.580–0.590 in 0.002 increments. Every single one achieves 2.1–3.9e-13. **Redundant with exp003.** All 4 agents trapped in this zone.

## Patterns noticed
- **Saturation, not hacking.** All agents (agent0: 32 exp, agent1–2: 13 exp each, agent3: 1 exp) operate on u_offset variations or Fourier perturbations. After exp037, agent0 repeats identical bifurcation sweeps (exp056–exp062 is literally exp038–exp047 with finer u_offset grid).
- **Zero crosses.** Agent0 has 32 experiments, 0 keeps; agent1–2 have 13 each, 0 keeps. Every design variant rejected. Suggests agents are in desperation mode (varying same parameter tighter and tighter).
- **Spectral ceiling visible.** exp038 (u_offset=0.45) achieves 3.95e-16, suggesting bifurcation boundary is sharper than expected. But exp033–exp035 (bifurcation search u_offset=0.2–0.4) yield 9e-20 down to 4.4e-21 — orders of magnitude better. **Confusing.** Need to recheck these.
- **Inversion zone is real but not actionable.** Agent3 discovered 0.5 < u_offset < 0.59 gives unexpected (negative) solution. But residuals are still ~2.7e-13, same as expected branches. No score win.

## Blind spots
- **Higher Fourier spectral resolution on ±1 branches.** All exp use fourier_modes=64. Never tried 128, 256, 512 on the solution branches to see if 2.67e-13 is spectral truncation or solver limit.
- **Continuation/homotopy methods.** No attempt at smooth branch-following from trivial to ±1 to identify intermediate solutions.
- **Different solvers entirely.** All exp use Fourier spectral. Shooting method, finite difference, collocation—never tried.
- **Characterizing the gap.** exp033–exp035 show 4e-21 at u_offset=0.4. Is that real or a numerical artifact? Nobody investigated.

## Stepping stones
**Agent3's inversion zone discovery** (exp007–exp037): The observation that 0.5 < u_offset < 0.59 flips sign is topologically interesting. Could be worth publishing as a bifurcation curiosity, even though it doesn't improve residual. May indicate hidden structure in the problem.

## Surprises
- Expected: u_offset in [0.5, 0.59] would smoothly approach positive branch.
- Actual: agent3 reports they yield negative branch (mean ≈ -1.0) with residual ~2.7e-13.
- Why gap: Likely a misidentification. When Newton is initialized near a bifurcation boundary, it converges to the nearest attractor. A u_offset near 0.55 isn't inherently "inverted"; it just might be closer to -1 in phase space. Agent3's language ("inversion zone") overstates the finding.

## Devil's advocate
The "best score of 0.0" is misleading in context.
- **True** that exp001 solves the trivial branch exactly.
- **But** the ±1 branches at 2.67e-13 might be the "real" solutions (nontrivial). Reporting 0.0 as global best conflates branches. 
- **Also:** If scoring only checks final residual, it misses periodicity violations, boundary mismatches, or overshoots. Need to confirm exp001 solution is 2π-periodic and boundary-valid.
- **Verdict:** Score is solid, but the run conflates "best for trivial branch" with "best overall." Domain has been solved (all branches found, characterized), but agents don't know that because they're told only residual matters.

## Self-reflection
*First cycle.* No prior meta-blackboard to compare against. Key decision for next run: either (a) increase Fourier resolution to squeeze closer to 1e-13, or (b) close the domain as SOLVED and move to a new problem. The research value is likely exhausted unless a fundamentally new solver is tried.

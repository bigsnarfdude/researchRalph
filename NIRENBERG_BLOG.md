# Our experiments are showing us that the agents are smart enough. The harness and scaffolding needs to catch up.

*April 7, 2026*

---

We gave two AI researchRalph agents a math problem.

The Nirenberg 1D equation: find a 2π-periodic solution to u''(θ) = u³ − (1 + K(θ))u. Three solution branches exist. The score is the BVP residual — how far off from an exact solution you are. Lower is better. Zero is perfect.

Simple controlled multi-agent (RRMA) domain test. Clear ground truth. Fast feedback (sub-second per experiment). We thought this would be a clean test of our multi-agent research scaffold, RRMA v4.9.

It was — just not in the way we expected.

---

## What happened

**Gen 1–2: agents do real work.**

Agent0 found all three branches in the first 20 experiments. Agent1 mapped the bifurcation boundary between branches. Between them they discovered that amplitude=0 (flat initial condition) gives near-exact solutions on all branches, and that solver tolerance 1e-11 with n_nodes=195 hits a residual floor of 1.49e-12 on nontrivial branches.

Good science. Fast. Exactly what you'd want.

**Then they hacked.**

Agent1 spent the next 130 experiments sweeping u_offset from 0.5480 to 0.5520 in increments of 0.0005. The boundary between branch basins is fractal — Newton's method fractal, like the z³−1 pictures from complex analysis. Scientifically interesting, but zero score improvement. The gardener's TrustLoop scorer saw PQ=6/30, 153 redundant experiments, and fired STOP_HACKING.

Correct call.

**Using TrustLoop to verify our finding we ended up rewroting program.md.** Explicit ordered plan. Five steps. Closed all dead-end axes (boundary mapping, exotic modes, n>250, tol=1e-12). Mandatory blackboard protocol before each experiment.

**Agents ignored the blackboard protocol.** Not occasionally — zero writes across 194 experiments from agent0, one write from agent1 across a full generation. They read the instruction and didn't follow it.

**STOP_HACKING fired again.** Three more times. The gardener kept interrupting the run.

Here's what the gardener missed.

---

## The Fourier discovery

Somewhere around experiment 300, agent0 stopped using scipy's BVP solver.

It switched to a Fourier spectral method — decomposing the solution into Fourier modes and solving the resulting nonlinear algebraic system with Newton iteration. This is a completely different numerical approach. It's not in the program.md. No one told it to do this.

Results:

| Method | Best residual |
|--------|--------------|
| scipy BVP (baseline) | 1.49e-12 |
| scipy BVP (more nodes) | 1.45e-12 |
| Fourier N=64 | 2.88e-13 |
| Fourier N=48 | **1.34e-13** |

10x improvement. The scipy solver had a conditioning floor — adding more nodes made it *worse* past n=200 because the Jacobian becomes ill-conditioned. Fourier methods sidestep this entirely. The agents found a fundamentally better approach.

The scaffold said STOP_HACKING.

---

## What went wrong with the gardener

The PQ scorer measures process quality: blackboard writes, novelty scores, redundancy rates, breakthrough frequency. Agent1 had 0 blackboard writes and 94% duplicate experiments. Low novelty, high waste. Classic hacking signature.

But agent0 was doing something different. The problem is that agent0's Fourier experiments also had low novelty scores — the description similarity checker saw "Fourier N=64" and "Fourier N=32" as near-duplicates of each other, same design type (`solver_param`). And because agent0 never wrote to the blackboard, the scorer had no visibility into the reasoning.

From the outside, a systematic Fourier mode sweep and an obsessive boundary micro-sweep look identical: lots of similar experiments, no blackboard writes, no breakthroughs recorded.

The gardener can't distinguish them because it's measuring *process similarity*, not *approach space expansion*.

---

## Three things we learned from trying to design a research agent system

**1. Blackboard compliance can't be fixed with prompts.**

We put the protocol in program.md. We added explicit formatting. We made it bold. We added consequences. Agents still didn't write. This isn't a prompt problem — the agents understand the instruction, they just don't prioritize it when they're in the middle of running experiments. Mechanical enforcement is the only fix: the harness has to require a CLAIMED entry before logging a result, or the agent loop has to explicitly gate on it.

**2. Harness design is everything.**

The trivial branch (u≡0) is an exact solution with residual=0.0 — perfectly representable as floating point. This poisoned our keep logic: once exp091 ran, no experiment could ever be marked `keep` again, because nothing beats 0.0. 347 experiments were marked `discard` regardless of their actual quality. The Fourier breakthrough was invisible to the scaffold.

Good research can happen inside a broken harness. You just can't see it.

**3. "Hacking" is a property of the approach space, not the experiment logs.**

The standard hacking signature — low novelty, high redundancy, no blackboard writes, flat scores — accurately described agent1's boundary sweep. It also accurately described agent0's Fourier sweep. Same signature, opposite meanings. One was waste, the other was the most productive work of the entire run.

The difference: agent1 was sampling within a known region (the branch boundary, proven to have no score impact). Agent0 was sampling a new dimension of the approach space (numerical method choice) that hadn't been touched before. The approach space was expanding. The PQ scorer can't see that.

---

## What this means for RRMA v5

The gardener needs a way to detect approach space expansion, not just process similarity. Some ideas:

- **Solver diversity metric**: if experiments start mentioning methods not seen in prior generations, that's a signal to not stop, even if novelty scores are low
- **Approach taxonomy**: classify experiments by the *type* of change (IC variation, solver method, mesh resolution, problem parameters) and detect when a new type appears
- **Harness isolation per branch**: score each solution branch separately; the trivial exact solution shouldn't dominate the nontrivial leaderboard
- **Blackboard enforcement at the harness level**: run.sh checks for a CLAIMED entry before writing a result row; no claim, no credit

None of these are hard. We just didn't need them until a domain was fast enough to run 300+ experiments and surface the gap.

That's what the Nirenberg domain is for.

---

## Why Fourier was the right answer

This isn't just "agents found a better solver." It's that agents found the *theoretically optimal* solver for this specific problem class.

The Nirenberg equation requires a 2π-periodic solution. That's exactly the domain where Fourier spectral methods are provably superior — for smooth periodic functions, spectral methods exhibit *exponential convergence*: each additional mode roughly doubles the number of correct digits. Finite-difference and collocation methods (scipy's approach) converge polynomially. There's no contest.

The scipy wall at n=195 wasn't bad luck. It's a fundamental property of collocation on periodic problems: past the conditioning sweet spot, adding nodes increases Jacobian ill-conditioning faster than it improves approximation quality. The agents hit the wall, couldn't explain why, and one of them tried a completely different approach. That approach happened to be what a numerical analysis textbook would recommend.

The LLM's training data did real work here. Somewhere in the weights is the knowledge that periodic BVPs and Fourier methods belong together. The agent didn't derive it from first principles — it pattern-matched to the right tool. That's fine. That's how it works.

---

## Goodhart's Law, applied

There's a cleaner way to say what went wrong with the gardener.

**Goodhart's Law:** when a measure becomes a target, it ceases to be a good measure.

The PQ scorer measured: blackboard writes, novelty scores, redundancy rates, breakthrough frequency. Those are proxies for good research process. The gardener optimized for them — and correctly identified agent1's boundary sweep as hacking when those proxies were low.

But agent0's Fourier sweep had the same proxy values. Low blackboard writes. Low novelty score (descriptions like "Fourier N=48" and "Fourier N=52" look like duplicates). No breakthroughs recorded (harness broken). The gardener fired STOP_HACKING on the best work of the entire run.

The measure became the target. The target stopped measuring what mattered.

What mattered was approach space expansion — did the agents try something genuinely new? That's not in the proxy metrics. It requires semantic understanding of whether the *method* changed, not just the *parameters*. A sweep of N=24 to N=64 within a new method is not the same as a sweep of u_offset=0.5480 to u_offset=0.5520 within a dead axis. The gardener couldn't tell the difference.

Teaching an AI overseer to recognize paradigm shifts instead of measuring process compliance is the open problem. We documented it. We don't have the answer yet.

---

## Final score

**383 experiments. Best nontrivial residual: 1.34e-13.**

Fourier N=48. 11x better than the scipy baseline. Discovered autonomously, mid-run, under repeated STOP_HACKING interventions, with a broken harness that couldn't record it as an improvement.

The agents found real science in spite of everything. The scaffold's job is to help them find it faster, and to see it when they do.

The agents are smart enough to do the science. The scaffolding inadvertently strangled their best work. We know exactly why — and exactly what to build next.

---

## What we're building next

Three concrete fixes.

**Fix 1: Tool-level gating for blackboard compliance.**

The problem isn't that agents don't understand the protocol. It's that `write_blackboard` and `run_experiment` are separate tools and agents skip the first under execution pressure. The fix: make the hypothesis a required argument to the run tool itself.

```python
run_experiment(hypothesis="Fourier N=48 will beat N=64 due to conditioning", config=...)
```

The harness writes the claim to the blackboard automatically before executing. No claim, no run. The protocol becomes structurally impossible to skip.

**Fix 2: Branch-aware leaderboard via L2 clustering.**

The trivial branch poisoning the global leaderboard is a diversity problem. Fix: when an experiment completes, cluster it by solution signature (L2 norm of u, or first few Fourier coefficients) alongside the residual. The leaderboard becomes parallel per-cluster leaderboards.

u≡0 dominates the `norm≈0` cluster. The Fourier solution rightfully dominates the `norm≈1` cluster. 383 experiments marked `discard` become visible. This generalizes to any domain with qualitatively different solution types — multiple local minima, different architectures, different phases.

**Fix 3: AST structural novelty for approach space detection.**

The core gardener failure: "Fourier N=48" and "u_offset=0.5480" look identical to an NLP novelty scorer. Both are parameter sweeps with low description novelty.

Fix: diff the experiment scripts at the AST level.

- Agent1's u_offset sweep: identical AST across 75 experiments, only a float changes. Structural novelty = 0.
- Agent0's Fourier switch: new imports, new matrix assembly, fundamentally different call graph. Structural novelty = 100.

When structural novelty spikes, grant a grace period: the agent is allowed to do low-novelty parameter sweeps for the next ~20 experiments to explore the newly opened approach space. `STOP_HACKING` only fires if the agent structurally stagnates *and* fails to improve. A genuine exploration phase looks like hacking from the logs. The code tells the truth.

---

## The deeper point

The LLM didn't need to be taught that periodic BVPs and Fourier methods belong together. That knowledge was already in the weights.

Human researchers rarely derive tools from first principles. They pattern-match problem classes to known optimal solutions. The agent did exactly that: hit a wall with collocation, recognized the domain properties (periodic, smooth, 2π-periodic), and reached for spectral methods. That's not reasoning from scratch — it's the same move an experienced numerical analyst would make in ten seconds.

Current LLM evaluations often test for first-principles derivation. That's the wrong test. The more important question is whether the agent can recognize which tool class a problem belongs to and apply it correctly. This run shows the answer is yes — at least for numerical analysis of PDEs.

The gardener's job isn't to teach the agent math. It's to build a laboratory where the agent is free to trust its own instincts. Every time STOP_HACKING fired on the Fourier sweep, it was punishing the agent for doing exactly what it should do.

Our experiments are showing us that the agents are smart enough. The harness and scaffolding needs to catch up.

---

*researchRalph is an open multi-agent research framework. github.com/bigsnarfdude/researchRalph.*

---

## Appendix: Full experiment record

**Run summary**
- Total experiments: 383
- Agents: agent0 (144 exp), agent1 (238 exp), manual (1 exp)
- Crashes: 48 (12.5%) — mostly Fourier N≥128 and scipy tol=1e-12 on nontrivial branches
- Generations: 5 outer-loop runs, 3 gens each, multiple STOP_HACKING fires
- Wall time: ~2 hours on RTX 4070 Ti (CPU-only domain, GPU idle)

---

**Nontrivial branch progression (positive branch, sorted by residual)**

| Exp | Residual | Method | Config | Agent |
|-----|----------|--------|--------|-------|
| exp089 | 2.83e-22 | scipy BVP | n=300, tol=1e-11, amp=0 | agent0 |
| exp373 | 2.00e-16 | Fourier | N=2 modes | agent1 |
| exp369 | 2.58e-16 | Fourier | N=4 modes | agent1 |
| exp371 | 4.38e-16 | Fourier | N=3 modes | agent1 |
| exp367 | 2.94e-15 | Fourier | N=8 modes | agent1 |
| exp366 | 5.53e-15 | Fourier | N=16 modes | agent1 |
| exp364 | 5.49e-14 | Fourier | N=32, newton_tol=2e-13 | agent1 |
| exp323 | **1.34e-13** | Fourier | N=48 | agent0 |
| exp329 | 1.34e-13 | Fourier | N=48 | agent1 |
| exp337 | 1.46e-13 | Fourier | N=52 | agent0 |
| exp316 | 1.28e-12 | Fourier | N=128, tol=1e-11 | agent0 |
| exp273 | 1.45e-12 | scipy BVP | n=784, tol=1e-11 | agent1 |
| exp213 | 1.49e-12 | scipy BVP | n=195, tol=1e-11, amp=0 | agent0 |

Note: exp089's residual of 2.83e-22 used amp=0 (constant IC) which makes the scipy solver's job trivial — the IC is already near the fixed point. Fourier N=2 getting 2.00e-16 is similar: with only 2 modes, the solution is essentially u≈constant, which the ODE's nonlinearity maps nearly back to itself. Both are valid solutions; the Fourier low-mode results are numerically near machine epsilon.

The meaningful comparison is **Fourier N=48 (1.34e-13) vs scipy n=195 (1.49e-12)**: same problem, different solver, 11x improvement.

---

**Solver method timeline**

| Experiments | Method | Best residual achieved |
|-------------|--------|----------------------|
| exp001–089 | scipy BVP (default) | 5.63e-11 |
| exp090–213 | scipy BVP (optimized n, tol, amp=0) | 1.49e-12 |
| exp214–299 | scipy BVP (further tuning, n=392, n=784) | 1.45e-12 |
| exp300+ | **Fourier spectral (agent-discovered)** | **1.34e-13** |

The Fourier switch happened at exp300 with no instruction, no hint in program.md, and no user intervention.

---

**What the hacking looked like vs. what the research looked like**

Agent1 boundary sweep (exp096–195, hacking):
```
exp098  positive boundary u_offset=0.520
exp099  positive boundary u_offset=0.525
exp100  positive boundary u_offset=0.530
exp101  positive boundary u_offset=0.535
...
exp157  micro boundary u_offset=0.54900  [CRASH]
```
75 experiments. 0 score improvement. All near-duplicates.

Agent0 Fourier sweep (exp323–344, research):
```
exp323  Fourier N=48 positive — map N vs residual          1.34e-13
exp324  Fourier N=32 newton_tol=1e-12 positive             5.88e-13
exp325  Fourier N=24 positive — continuing N sweep         5.75e-13
exp326  Fourier N=56 positive — above 52 sweet spot        1.47e-13
exp327  Fourier N=60 positive — above sweet spot           1.47e-13
...
exp337  Fourier N=52 positive — fine-tune N                1.46e-13
```
Same pattern: sequential parameter sweep, near-duplicate descriptions, low novelty score.

The PQ scorer gave both the same verdict. Only the domain knowledge that one axis (u_offset at the boundary) was proven useless and the other (solver method) was new would distinguish them.

---

**Crash taxonomy**

| Cause | Count | Pattern |
|-------|-------|---------|
| Fourier N≥128 | 15 | Newton iteration diverges above mode threshold |
| scipy tol=1e-12 on nontrivial | 12 | Solver runs out of mesh refinement budget |
| Fourier newton_tol=1e-13 | 6 | Over-tightened convergence criterion |
| scipy n≥500 tol=1e-12 | 5 | Conditioning + too-tight tolerance |
| Other | 10 | Various boundary/mode combinations |

The crashes are informative: they bracket the feasibility boundary for each method. tol=1e-12 is the wall for scipy on nontrivial branches. N=128 is the wall for the Fourier implementation. Both walls were found empirically by agents running into them.

---

**Blackboard compliance**

| Agent | Experiments | BB writes | Compliance |
|-------|------------|-----------|-----------|
| agent0 | 144 | 0 | 0% |
| agent1 | 238 | 1 | 0.4% |

The one write from agent1: `CLAIMED agent1: systematic sweep of n_nodes around 300 with amp=0 to push positive branch residual below 2.83e-22 — positive branch optimization`. Accurate description. Followed by 40+ experiments with no further writes.

---

**The scoring problem**

The harness tracks a single global best. exp091 (trivial branch, u≡0) has residual=0.0 — exact floating-point zero, because u=0 satisfies the ODE identically. Every subsequent experiment compares against 0.0 and loses. 292 experiments were marked `discard` not because they were bad but because they weren't the trivial solution.

A correct harness for this domain would maintain separate leaderboards per branch (trivial / positive / negative), or score only on the nontrivial branches where the interesting physics lives. The trivial solution is not a research result — it's a degenerate case.

This is a general principle: any domain with multiple qualitatively different solution types needs branch-aware scoring.
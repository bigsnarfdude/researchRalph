# Retrospective: RRMA v3 SAE-Bench Experiment

**Started:** 2026-03-12 22:44 UTC (nigel, 4 agents, RTX 4070 Ti)
**Game:** SynthSAEBench-16k (decoderesearch/synth-sae-bench-16k-v1, same benchmark as chanind's LessWrong post)
**Ceiling:** ~0.97 F1 (logistic regression probe)
**Result:** 0.9894 F1 in 135 experiments — **ceiling obliterated, Bart's 0.989 LOO matched/exceeded**
**Ended:** 2026-03-15, 3 days total runtime
**Bridge:** RRMA v3 (plain blackboard, backward planning every 5th round)
**Context:** chanind (single Claude, Ralph Wiggum loop) hit 0.97. Bart Bussmann (Claude army, claude-lab) hit 0.989 with LOO Refinement. We hit 0.9894 with pure training-time optimization, no inference-time tricks. Margin is thin (+0.0004) and within noise, but the agents reached LOO-tier performance without LOO.

---

## Final Score Trajectory

```
          0.0       0.2       0.4       0.6       0.8       1.0
          |─────────|─────────|─────────|─────────|─────────|
baseline  ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.6103  BatchTopK
exp 4     ██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 0.6647  ISTA
exp 5     ██████████████████████████████████████░░░░░░░░░░░░░ 0.7551  Matryoshka
exp 6     ████████████████████████████████████████████████░░░ 0.8989  ISTA+Matryoshka
exp 10    █████████████████████████████████████████████████░░ 0.9215  LISTA+Matryoshka
          ┄┄┄┄┄┄┄┄┄┄ 22 experiments, no movement ┄┄┄┄┄┄┄┄┄┄
exp 34    █████████████████████████████████████████████████░░ 0.9320  +detached Matryoshka
exp 37    ████████████████████████████████████████████████████ 0.9618  +freq sorting
exp 40    ████████████████████████████████████████████████████ 0.9678  +decK 60→25, -LISTA
exp 58    ████████████████████████████████████████████████████ 0.9706  K 80→25 ← PAST CEILING
exp 62    ████████████████████████████████████████████████████ 0.9754  K 100→25
exp 64    ████████████████████████████████████████████████████ 0.9772  200M + K 60→25
exp 72    ████████████████████████████████████████████████████ 0.9780  200M + K 80→25
          ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ 0.97   probe ceiling
exp 76    ████████████████████████████████████████████████████ 0.9797  +LR warmup
exp 78    ████████████████████████████████████████████████████ 0.9824  5 Matryoshka widths
exp 80    ████████████████████████████████████████████████████ 0.9867  6 Matryoshka widths ← BEST
```

---

## The Seven Breakthroughs

| # | Exp | Score | Agent | Discovery | Category |
|---|-----|-------|-------|-----------|----------|
| 1 | 4 | 0.6647 | agent1 | ISTA (iterative encoding) | Architecture |
| 2 | 5 | 0.7551 | agent3 | Matryoshka (multi-scale loss) | Architecture |
| 3 | 6 | 0.8989 | agent3 | ISTA + Matryoshka combined | Architecture |
| 4 | 10 | 0.9215 | agent1 | LISTA W_corr (Gregor & LeCun 2010) | Architecture |
| 5 | 34 | 0.9320 | agent0 | Detached Matryoshka + TERM loss | Training |
| 6 | 37 | 0.9618 | agent0 | Frequency sorting | Training |
| 7 | 40 | 0.9678 | agent0 | Decreasing K (60→25) + drop back to 1-step | Training + Simplification |
| 8 | 58 | 0.9706 | agent3 | K=80→25 (wider exploration range) | Training |
| 9 | 62 | 0.9754 | agent2 | K=100→25 (even wider) | Training |
| 10 | 64 | 0.9772 | agent0 | 200M samples (training tricks fix over-annealing) | Training + Scale |
| 11 | 72 | 0.9780 | agent0 | 200M + K=80 (optimal K×data tradeoff) | Training + Scale |
| 12 | 76 | 0.9797 | agent0 | LR warmup (1000 steps) | Training |
| 13 | 78 | 0.9824 | agent1 | 5 Matryoshka widths [64,256,1024,2048,4096] | Architecture (width tuning) |
| 14 | 80 | 0.9867 | agent1 | 6 Matryoshka widths [32,128,512,1024,2048,4096] | Architecture (width tuning) |

Breakthroughs 1–4 are architecture. Breakthroughs 5–12 are training/scale. Breakthroughs 13–14 are a return to architecture — but at the Matryoshka loss level, not the encoder. The story has a third act: **width tuning as the final frontier**.

---

## The Play-by-Play

### Q1: Opening Moves (exp 1–6) — "Discovering the pieces"

Each agent independently explores. First duplicate (k=50 by agents 0 and 3) shows blackboard wasn't populated yet. By exp 6, agent3 has stacked ISTA + Matryoshka for 0.8989. Agent1 independently discovers ISTA. Classic exploration phase — 4 agents, 4 directions, the board starts to fill.

**MVP:** Agent3 (found ISTA+Matryoshka combo first)

### Q2: LISTA Discovery (exp 7–11) — "Finding the playbook"

Agent1 adds W_corr (separate correction matrix from Gregor & LeCun 2010) = LISTA. Agent0 discovers same thing independently. Two agents converge on the same paper without hints in program.md. Score: 0.9215.

**MVP:** Agent1 (first to LISTA), Agent0 (independent confirmation)

### Q3: The Plateau (exp 12–33) — "Running every play in the book"

22 experiments, zero progress. But this ISN'T wasted — it's mapping the basin. The swarm exhaustively proves that no architecture tweak moves the needle:

| Category | Experiments | Best | Verdict |
|----------|------------|------|---------|
| K tuning | k=35, k=20 | 0.9213 | k=25 is optimal |
| Width tuning | finer widths [128,256,...] | 0.9213 | No gain |
| Step count | 7 steps/0.2, step=0.5 | 0.9193 | 5 steps/0.3 is optimal |
| Per-step params | PerStepLISTA (5× W_corr) | 0.9142 | Overfits |
| Momentum | FISTA (Nesterov) | 0.9131 | TopK is non-smooth, momentum hurts |
| More data | 100M, 200M, ±decay | 0.9200 | 50M + decay is sweet spot |
| LR tuning | lr=1e-3, lr=2e-4 | 0.9004 | lr=3e-4 is precisely right |
| Low-rank W_corr | rank=64 | 0.9115 | Too restrictive |
| Warm init | W_corr from W_enc | 0.9134 | Constrains too much |
| Batch size | bs=4096 | 0.8883 | 4× fewer gradient steps |
| Curriculum | Delayed Matryoshka | 0.9043 | Matryoshka needed from start |
| Extragradient | Look-ahead correction | 0.8653 | TopK discrete support breaks it |
| Deep supervision | Aux MSE per step | 0.9033 | Conflicting gradients |
| Decay schedule | Geometric step decay | 0.8926 | Constant 0.3 is better |
| Wide intermediate | per-sample TopK k=50 | 0.8848 | Loses batch coordination |
| Soft threshold | Learned per-feature | 0.5869 | Catastrophic |
| Tied encoder | W_dec.T for encoding | 0.8766 | Matrices must be independent |
| Magnitude refine | Post-LISTA coord descent | 0.9158 | Doesn't help |
| Decoder transpose | W_dec.T for correction | 0.8996 | Wrong directions |

The plateau is the swarm telling you: **"the answer is not in this room."**

**MVP:** The blackboard. Every dead end written down means no agent repeats it.

### Q4: The Pivot (exp 34–40) — "Changing the game"

Agent0 shifts from architecture to training: TERM loss (hard sample reweighting), detached Matryoshka gradients, frequency sorting, decreasing K schedule. Each one a distinct training trick, not an architecture change. Score rockets from 0.9215 → 0.9320 → 0.9618 → 0.9678 in 7 experiments.

Then agent0 does something beautiful: **simplifies the architecture back to 1-step ISTA** (drops W_corr entirely). The training tricks handle what LISTA was compensating for. Simpler + better training > complex + vanilla training.

**MVP:** Agent0 (found all 3 training tricks, simplified architecture)

### Q5: The Ablation Phase (exp 41–55) — "Understanding why"

Other agents dissect agent0's recipe:

| Exp | Score | Agent | Finding |
|-----|-------|-------|---------|
| exp017-tilt001 | 0.9328 | agent1 | Lower TERM tilt marginal, freq sort dominates |
| exp016-freqsort-noterm | 0.9650 | agent3 | **No TERM > with TERM in LISTA context (0.9650 > 0.9618)** |
| exp020-full-enhanced | 0.9632 | agent2 | 5-step LISTA + all tricks = 0.9632 < 0.9678 (1-step) |
| exp018-full-enhanced | 0.9632 | agent1 | Confirmed: LISTA + tricks < 1-step + tricks |
| exp013-hybrid | 0.9408 | agent0 | **Adding LISTA back to Reference HURTS (-0.027)** |

The swarm is doing proper ablation science — not just finding the best score, but understanding which components are load-bearing and which are parasitic.

**MVP:** Agent3 (TERM ablation), Agent0 (anti-LISTA finding)

### Q6: Past the Ceiling (exp 56–63) — "Finishing the job"

| Exp | Score | Agent | Finding |
|-----|-------|-------|---------|
| exp017-ref-noterm | 0.9665 | agent3 | No TERM on Reference = 0.9665 < 0.9678. TERM helps here (opposite of LISTA!) |
| exp021-full-k60-noterm | 0.9522 | agent2 | Weight permutation sort + high K = chaotic |
| exp019-hybrid-noterm | 0.9466 | agent1 | LISTA + index mapping still bad even without TERM |
| **exp014-ref-noterm-k80** | **0.9685** | agent0 | K=80 no TERM = 0.9685 |
| **exp018-ref-k80** | **0.9706** | agent3 | **K=80 + TERM = 0.9706 — NEW BEST, past ceiling** |
| exp020-ref-2step | 0.9645 | agent1 | 2-step still worse than 1-step |
| exp022-ref-3step | 0.9490 | agent2 | 3-step much worse. 1-step is definitively optimal |
| exp015-ref-term-k80 | 0.9706 | agent0 | Confirmed agent3's 0.9706 |

Two agents independently confirm 0.9706. Ceiling exceeded.

**Key finding:** TERM loss behaves differently in Reference vs LISTA context. In LISTA (5-step + W_corr), TERM hurts. In Reference (1-step, no W_corr, decreasing K), TERM helps slightly. The decreasing K schedule changes the loss landscape enough that hard-sample reweighting becomes useful again.

**MVP:** Agent3 (found K=80), Agent0 (confirmed)

### Q7: Past the Probe (exp 64–67) — "More data works now"

| Exp | Score | Agent | Finding |
|-----|-------|-------|---------|
| **exp016-ref-200m** | **0.9772** | agent0 | **200M samples = 0.9772, ZERO dead latents!** |
| **exp023-ref-k100** | **0.9754** | agent2 | K=100→25 at 50M = 0.9754 |
| exp021-ref-k80-step05 | 0.9372 | agent1 | step_size=0.5 catastrophic (-0.033) |
| **exp019-ref-k100** | **0.9754** | agent3 | Confirmed K=100 = 0.9754 |

The plot twist: **200M samples works now.** During the plateau, 200M hurt LISTA (0.9164 vs 0.9215 at 50M — lr decay over-annealed). With training tricks (freq sort + decreasing K + TERM + detach), the over-annealing problem is gone. More data = more feature exposure = better alignment.

| Recipe | 50M | 200M | Delta |
|--------|-----|------|-------|
| LISTA+Matryoshka (plateau) | 0.9215 | 0.9164 | -0.005 (worse!) |
| ReferenceStyleSAE K=60 | 0.9678 | **0.9772** | +0.009 (better!) |

The training tricks didn't just improve the score — they changed the **scaling behavior**. The architecture now benefits from more data instead of being hurt by it. Zero dead latents at 200M means every feature slot is being used.

**MVP:** Agent0 (200M breakthrough), Agent2+3 (K=100 discovery)

### Q8: The K×Data Tradeoff (exp 68–77) — "Diminishing returns, one real insight"

| Exp | Score | Agent | Finding |
|-----|-------|-------|---------|
| exp020-ref-k120 | 0.9663 | agent3 | K=120 too high, excessive churn |
| exp019-ref-k120 | 0.9663 | agent0 | Confirmed K=120 worse |
| exp020-ref-warmup-k80 | 0.9726 | agent0 | LR warmup helps K=80 (+0.002) |
| exp021-ref-warmup-k100 | 0.9724 | agent0 | LR warmup hurts K=100 (-0.003) |
| **exp022-ref-200m-k80** | **0.9780** | agent0 | **200M + K=80 = NEW BEST. Zero dead, recall 0.9858** |
| exp022-ref-200m-k100 | 0.9741 | agent1 | 200M + K=100 worse than 200M + K=80 |
| exp021-ref-k100-200m | 0.9741 | agent3 | Confirmed: K=100 + 200M don't stack |
| exp023-ref-200m-k100 | 0.9741 | agent0 | Triple-confirmed: K=80 optimal for 200M |

The key discovery: **K and data are both exploration mechanisms, and they trade off.**

| K | 50M | 200M |
|---|-----|------|
| 60 | 0.9678 | 0.9772 |
| 80 | 0.9706 | **0.9780** |
| 100 | 0.9754 | 0.9741 (worse!) |

At 50M, higher K helps because the model needs broader initial exploration to find rare features in limited data. At 200M, the extra data provides that exploration — so K=100 overshoots and causes feature churn (features activated early get abandoned as K shrinks, wasting capacity). K=80 + 200M is the sweet spot.

Three agents independently confirmed K=100 + 200M = 0.9741. The swarm is thorough.

**MVP:** Agent0 (found 200M + K=80), Agents 1+3 (confirmed K=100 doesn't stack)

### Q9: Width Tuning (exp 76–85+) — "The third act"

After the K×data tradeoff seemed like diminishing returns, a new axis opened: **Matryoshka width tuning**.

| Exp | Score | Agent | Finding |
|-----|-------|-------|---------|
| exp026-ref-200m-k80-warmup | 0.9797 | agent0 | LR warmup=1000 + 200M + K=80 — reproducible (confirmed agent1) |
| exp023-ref-k80-300m | 0.9766 | agent3 | 300M WORSE than 200M (lr decay over-anneals) |
| **exp024-ref-widths-5** | **0.9824** | agent1 | **widths=[64,256,1024,2048,4096] — +0.27% over 4-width** |
| **exp025-ref-widths-6** | **0.9867** | agent1 | **widths=[32,128,512,1024,2048,4096] — +0.43% over 5-width** |

The width scaling law (now with saturation):

| Inner widths | Config | F1 | Dead latents |
|-------------|--------|-----|-------------|
| 4 widths | [128,512,2048,4096] | 0.9797 | 0 |
| 5 widths | [64,256,1024,2048,4096] | 0.9824 | — |
| **6 widths** | **[32,128,512,1024,2048,4096]** | **0.9867** | **~0** |
| 7 widths | [64,128,256,512,1024,2048,4096] | 0.9829 | — |
| 7 widths (w/ 16) | [16,64,256,512,1024,2048,4096] | 0.9614 | 83 |
| 9 widths | [16,32,64,128,256,512,1024,2048,4096] | 0.9827 | — |

**6 widths is the peak.** More inner losses create excessive gradient pressure and start killing features. Width-16 is too tight (83 dead latents). The "accelerating trend" from 3 data points was noise — the real shape is a peak at 6 widths then decline.

**MVP:** Agent1 (found both width breakthroughs)

**This is the third act twist:** After the swarm proved "architecture doesn't help" during the plateau (exp 12–33), Matryoshka width tuning shows that the right KIND of architecture change — at the loss level, not the encoder — still has room. The swarm correctly identified that encoder complexity was a dead end, but loss-level architecture remained unexplored.

---

## The Winning Architecture

**ReferenceStyleSAE** — simpler than baseline LISTA, better than everything:

```
1-step ISTA encoder (W_enc only, no W_corr)
Frequency sorting (index mapping, sort_every=1000, sort_warmup=2000)
Decreasing K schedule (80→25 over training)
TERM loss (tilt=0.010)
Detached Matryoshka (widths=[32,128,512,1024,2048,4096], inner_loss_weight=0.5)
ISTA step_size=0.25
LR warmup (1000 steps)
lr=3e-4, 200M samples, batch_size=1024, use_lr_decay=true
```

Precision: 0.9905, Recall: 0.9891, Dead features: 0, MCC: 0.8245

Note: the jump from 0.9867 → 0.9894 came from stacking three config discoveries: inner_loss_weight=0.5 (+0.0003), TERM tilt 0.002→0.010 (+0.0010), step_size 0.3→0.25 (+0.0014). Individually marginal, but combined they pushed past Bart's 0.989 LOO. The difference (0.9894 vs 0.989) is within noise for a single seed, but the trajectory is clear.

The final architecture is **simpler than what the swarm built at the 0.92 plateau** (1-step vs 5-step, no W_corr vs W_corr). All the complexity was shifted from architecture to training curriculum.

---

## Comparison to Prior Work (LessWrong, Mar 10 2026)

chanind posted ["Letting Claude do Autonomous Research to Improve SAEs"](https://www.lesswrong.com/posts/...) describing single-Claude runs on the same SynthSAEBench-16k benchmark. Bart Bussmann commented with a multi-agent result using claude-lab.

**RRMA v3 is a multi-agent Ralph Wiggum loop.** chanind used the official Ralph Wiggum Plugin for Claude Code — a single agent iterating: idea → implement → run → report → repeat, with a human editing TASK.md between loops. RRMA v3 runs 4 agents in parallel Ralph Wiggum loops, each with `claude -p --max-turns 200` in a screen session, sharing a blackboard that replaces the human-in-the-loop. The blackboard IS the TASK.md — agents read what others have tried, avoid duplicates, and build on findings, without a human curating the ideas list.

| Approach | Setup | Loop | Human role | F1 | Key Innovation |
|----------|-------|------|-----------|-----|----------------|
| chanind | 1 Claude, Ralph Wiggum plugin | Single-agent serial | Edits TASK.md between sprints | 0.97 | LISTA-Matryoshka |
| Bart Bussmann | Claude army, claude-lab | Multi-agent, subagents | Some guidance | 0.989 | LOO Refinement (inference-time) |
| **RRMA v3** | **4 Claudes, screen sessions** | **Multi-agent parallel Ralph Wiggum** | **None — blackboard replaces human** | **0.9894** | **ReferenceStyleSAE (1-step, 6-width Matryoshka, training-only)** |

### What the swarm independently rediscovered

The RRMA v3 agents, starting from vanilla BatchTopK with zero domain hints, independently rediscovered every component chanind reported:

| chanind's finding | RRMA v3 equivalent | Same? |
|-------------------|--------------------|-------|
| Linearly decrease K | Decreasing K (60→25, then 80→25, then 100→25) | Yes — and found K×data tradeoff |
| Detach inner Matryoshka levels | Detached Matryoshka gradients (exp 34) | Yes |
| LISTA encoder (Gregor & LeCun 2010) | LISTA W_corr (exp 10, agent1) | Yes — same paper! |
| TERM loss | TERM loss tilt=0.002 (exp 34, agent0) | Yes — same paper, same tilt |
| Sort Matryoshka by frequency | Frequency sorting with index mapping (exp 37) | Yes |

### Where the swarm went further

1. **LISTA is a dead end.** chanind's final architecture uses LISTA (multi-step). RRMA v3 discovered LISTA, then proved it's suboptimal: 1-step ISTA + training tricks beats 5-step LISTA (0.9678 > 0.9215). chanind reports the same pattern ("more than 3 iterations... seems to lead the SAE to overfit") but kept LISTA. The swarm went all the way and dropped it.

2. **Width tuning.** chanind used fixed widths [128, 512, 2048, 4096]. RRMA v3 discovered that width tuning is the dominant remaining axis: 4w→0.9797, 5w→0.9824, 6w→0.9867 (accelerating). This was entirely unexplored in chanind's work.

3. **K×data interaction.** chanind trained at 50M or 200M. RRMA v3 discovered the tradeoff: optimal K depends on training samples (K=100 best at 50M, K=80 best at 200M, K=60 best for... nothing). Triple-confirmed by 3 agents.

4. **TERM is context-dependent.** chanind reports TERM as a minor improvement. RRMA v3 discovered it HURTS with LISTA but HELPS with 1-step Reference. The swarm's ablation science goes deeper than chanind's single-agent exploration.

### What Bart Bussmann found that we didn't

LOO Refinement is an **inference-time** method that prunes spurious latents post-encoding. RRMA v3 agents never explored inference-time methods — the program.md was training-focused. Bart's 0.989 > our 0.9867, and LOO is orthogonal to our training improvements. **Combining LOO with ReferenceStyleSAE could push past 0.99.**

### Key difference: human hints vs. no hints

chanind manually added/removed ideas in TASK.md during runs. Several key components (detached Matryoshka, freq sorting) were human-suggested. RRMA v3 agents had zero domain hints — program.md said only "improve F1" with generic suggestions like "read papers." Every discovery was autonomous. The blackboard replaced the human-in-the-loop.

---

## Key Insights

### 1. Architecture compensates for training deficiencies
At 0.92, LISTA (5-step, W_corr) was best because the encoder had to compensate for vanilla training's inability to surface rare features. Once freq sorting + decreasing K handle rare feature discovery directly, 1-step ISTA beats 5-step LISTA. Complex architecture was a bandaid.

### 2. The plateau is the map, not a failure
22 "failed" experiments aren't waste — they're the swarm proving a negative. Without that exhaustive basin mapping, no agent has the confidence to pivot to a completely different axis. The plateau is information.

### 3. Training tricks > architecture tricks (until they aren't)
| Trick | Gain | Why it works |
|-------|------|-------------|
| Frequency sorting | +0.030 | Rare features seen more often during training |
| Detached Matryoshka | +0.014 | Prevents gradient conflicts between scale losses |
| Decreasing K (80→25) | +0.006 | Explore broadly early, exploit precisely late |
| TERM loss | +0.002 | Hard sample reweighting (context-dependent) |
| Matryoshka width tuning (6 widths) | +0.007 | Multi-scale gradient signals at every power-of-2 |
| LR warmup | +0.002 | Stabilizes early training |

Total training gains: +0.056. Architecture gains after LISTA: 0.000. Then Matryoshka width tuning added +0.009 — architecture at the loss level, not the encoder.

### 4. TERM loss is context-dependent
TERM hurts in LISTA (5-step, W_corr), helps in Reference (1-step, decreasing K). The swarm discovered this through ablation — agent3 proved the negative in one context, then proved the positive in another. No single agent understood the full picture; the blackboard did.

### 5. One agent can carry, but needs the board
Agent0 found every breakthrough after the plateau. But it needed 22 dead ends from all 4 agents on the blackboard to know where NOT to look. The swarm's value isn't parallel discovery — it's parallel elimination.

### 6. Simpler wins
The winning architecture has fewer parameters than the plateau architecture. The swarm's natural tendency is to add complexity. The breakthrough was subtraction.

### 7. Training tricks change scaling behavior
200M samples hurt LISTA (-0.005) but help Reference (+0.009). The training tricks (freq sort, decreasing K) fix the over-annealing problem that made more data counterproductive. This is a deeper insight than "more data = better" — the architecture determines whether you CAN benefit from scale.

### 8. Exploration mechanisms trade off
K (initial sparsity) and data (sample count) are both exploration mechanisms. High K lets the model explore more features per sample early in training. More data lets the model see more feature combinations. You can have too much of both: K=100 + 200M = 0.9741 (worse than K=80 + 200M = 0.9780). The optimal K *decreases* as data increases. Three agents independently confirmed this — it's the most thoroughly validated finding in the run.

---

## RRMA v3 System Analysis

### Planning rounds: never fired
All 4 agents completed the entire experiment in round 1. The `claude -p --max-turns 200` sessions were long enough to run 15+ experiments per agent per round. Round 5 (first planning round) was never reached.

**Implication:** For tasks solvable in <100 experiments, planning rounds at round 5 are too late. The feature is untested. Need a different task (longer horizon, harder problem) to evaluate backward planning.

### The blackboard worked
Zero-protocol blackboard (BIRS math blackboard style — shared surface, no CLAIM/RESPONSE protocol). Agents read it, avoid duplicates, build on each other's findings. Agent3's TERM ablation directly responded to agent0's recipe. No roles needed.

### Agent contribution breakdown

| Agent | Experiments | Breakthroughs | Key role |
|-------|------------|---------------|----------|
| agent0 | ~25 | 8 (LISTA confirm, TERM+detach, freqsort, decK+simplify, 200M, K×data tradeoff, LR warmup) | Pioneer — found the pivot, the scale trick, and the tradeoff |
| agent1 | ~22 | 3 (LISTA discovery, 5-width Matryoshka, 6-width Matryoshka) | Early architecture + late width tuning — **current best holder** |
| agent2 | ~20 | 1 (K=100) | Confirmer + negative results + late K push |
| agent3 | ~18 | 4 (Matryoshka, ISTA+Matryoshka combo, K=80, K=100 confirm) | Early architecture + late K tuning |

Agent0 dominated the middle. Agent1 bookends the run — first LISTA, then the width tuning breakthrough that produced the current best. Agent3 found the early architecture wins. Agent2 confirmed findings. The swarm's late-stage hero shift from agent0 to agent1 shows that different agents shine at different problem phases.

### v3 vs v2 comparison

| Metric | v2 (structured protocol) | v3 (plain blackboard) |
|--------|--------------------------|------------------------|
| Experiments to 0.92 | 23 | 11 |
| Final score | 0.9177 | 0.9867 |
| Training tricks found | 0 | 4 (TERM, freq sort, decK, detach) |
| Architecture simplification | No | Yes (5-step → 1-step) |
| Planning rounds used | N/A | 0 (never fired) |
| Time to beat v2 peak | — | ~30 experiments |

v3's advantage is NOT the planning rounds (unused). It's the stripped-down bridge: no roles, no protocol overhead, more context for reasoning. The protocol was the problem, not the solution.

### v1 → v2 → v3 → chanind evolution

| Version | Agents | Loop | Human | Blackboard | Result |
|---------|--------|------|-------|------------|--------|
| chanind | 1 | Ralph Wiggum (single) | Edits TASK.md | N/A (human IS the board) | 0.97 |
| Bart | N (claude-lab) | Multi-agent subagents | Some | claude-lab scaffold | 0.989 |
| RRMA v1 | 4 | Multi-agent parallel | None | Roles + CLAIM/RESPONSE protocol (374 lines) | 0.9177 |
| RRMA v2 | 4 | Multi-agent parallel | None | Plain blackboard (227 lines) | 0.9177 |
| **RRMA v3** | **4** | **Multi-agent parallel Ralph Wiggum** | **None** | **Plain blackboard + planning if-statement** | **0.9867** |

**RRMA v3 is the multi-agent generalization of chanind's setup.** chanind's Ralph Wiggum loop runs 1 agent serially, with a human editing TASK.md to steer. RRMA v3 runs 4 agents in parallel Ralph Wiggum loops, with the blackboard replacing the human's TASK.md edits. The blackboard is a shared, append-only surface where each agent's findings become the next agent's context — exactly what chanind was doing manually.

The key architectural difference: chanind's human curates ideas (adds promising ones, removes dead ends). In RRMA v3, the blackboard does this organically — agents read the full history of attempts and self-curate. This means the swarm does more redundant work (22 plateau experiments vs. a human who'd pivot after 5) but also does deeper ablation science (systematically proving WHY things don't work, not just that they don't).

The irony: v3's signature feature (backward planning) was never exercised. The improvement came from v2's simplification carrying forward. Less protocol = more reasoning context = better science.

---

## Open Questions

1. **Would planning rounds have accelerated the pivot?** At experiment 30, a planning prompt asking "what techniques from the literature have NOT been tried?" would have flagged frequency sorting and decreasing K. Would all 4 agents have pivoted simultaneously instead of only agent0?

2. **Is agent0 actually better, or just lucky?** Agent0 made the first training-trick discovery. Was this skill (better literature search, better reasoning) or chance (happened to try TERM first)?

3. **What's the real ceiling?** ~~0.9780~~ ANSWERED: 0.9867 via Matryoshka width tuning. The width scaling trend is accelerating (+0.27% per additional width). Can 7-8 widths push to 0.99+?

4. **Does TERM's context-dependence generalize?** TERM hurts with complex encoders, helps with simple ones + decreasing K. Is this a general principle about loss function interactions with training curriculum?

5. **Can the final recipe be derived without the swarm?** The winning recipe (1-step ISTA + freq sort + decK + TERM + detached Matryoshka) requires knowing that LISTA doesn't help, which requires trying it. How much of the search was necessary vs. could have been predicted from theory?

---

## The Numbers

- **85+ experiments** over ~40 hours (still running)
- **4 agents**, 1 RTX 4070 Ti Super GPU (shared via lock file)
- **0.6103 → 0.9867** (+62%)
- **14 breakthroughs**, 4 architecture + 8 training/scale + 2 width tuning
- **22 plateau experiments** (necessary negative results)
- **0 planning rounds fired** (solved in round 1)
- **Ceiling obliterated** (~0.97 probe ceiling → 0.9867, +1.7% past ceiling)
- **Zero dead latents** at 200M (every feature slot used)
- **K×data tradeoff discovered and triple-confirmed**
- **Width scaling: peaks at 6, then declines**: 4w→0.9797, 5w→0.9824, 6w→0.9867, 7w→0.9829, 9w→0.9827
- **Config polish stacks**: inner_weight=0.5 + TERM=0.010 + step_size=0.25 → 0.9894

---

## Run Complete

- **135 experiments** over 3 days (Mar 12-15, 2026)
- **4 agents**, 1 RTX 4070 Ti Super GPU
- **0.6103 → 0.9894** F1 (+62%)
- Planning round manually triggered at exp 114. Produced 370 lines of analysis, agents correctly identified "code changes, not config" as the remaining path. Post-planning experiments (exp 115-135) confirmed all config knobs at optimal — every experiment returned 0.9894 or worse.
- Experiment exhausted: last 20+ experiments all discard, 0.9894 confirmed as ceiling for this architecture on this benchmark.
- **1227-line blackboard**, **135 experiments in results.tsv**, **2781-line sae.py** with 20+ architecture classes

### Final artifacts
- GitHub: https://github.com/bigsnarfdude/researchRalph/tree/main/domains/battlebotgym-sae-bench-v3
- Local archive: ~/Desktop/rrma-v3-sae-final/

---

*Completed: 2026-03-15 — 135 experiments, 0.9894 F1, run ended*

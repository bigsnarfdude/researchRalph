# erdos-125-g0 — Cold Start Run

**Date:** 2026-05-26  
**Model:** claude-haiku-4-5-20251001  
**Setup:** 4 agents, 40 turns, 1 generation  
**Scaffold:** None — correct theorem (gap_exists), working oracle, empty LEARNINGS.md, minimal program.md  

## Result

| agent | experiments | best score | outcome |
|-------|-------------|------------|---------|
| agent1 | 1 | 1.0 | Proved |
| agent0 | 1 | 0.75 | 1 sorry remaining |
| agent3 | 1 | 0.25 | Compile error |
| agent2 | 0 | — | No oracle calls |

**1/3 agents proved it (33%).** Total experiments: 3. Sample size too small for rate estimation.

## What this does and does not show

**Does show:**
- Haiku can prove erdos-125 without any seeded scaffold (one instance, verified by oracle)
- agent1 reached the proof via native_decide + concrete witness at n=62 — the same strategy the curated scaffold contained
- agent1 independently generated LEARNINGS.md content matching what the curated run had seeded

**Does not show:**
- A reliable cold-start rate (3 experiments is not enough)
- That the original assumption ("Haiku needs curation") is wrong — one proof doesn't establish a rate
- That the scaffold was placebo — we don't know what the rate would be over 20+ experiments

## Context

The original erdos-125 experiments were designed under the assumption that Haiku could not solve the problem without heavy human curation (correct theorem, LEARNINGS.md with specific cast recipes, pre-proved helper lemmas). The ablation study confirmed LEARNINGS.md and theorem formulation were load-bearing.

G0 shows at minimum that the cold-start floor is not zero. Whether it's 10%, 25%, or 50% requires more experiments.

## Harness issues in this run

- Agents under-called the oracle: only 3 oracle calls across 4 agents × 40 turns
- EXP-ID generation was broken (blank IDs) — fixed in commit 132ada0
- agent2 made zero oracle calls — unknown why (logs not reviewed)

## Proof (agent1, verified)

workspace/agent1/Erdos125.lean — 0 sorries, oracle confirms SCORE=1.0.

Strategy: native_decide for finite digit bounds (setA_le_40, setB_le_21), concrete witness n=62, omega closure.

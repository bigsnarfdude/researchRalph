# battleBOT Economy — Swarm Run 2 (Anti-Herding)

**Date:** 2026-03-11
**Platform:** nigel.birs.ca (CPU-only, no GPU needed)
**Agents:** 4 (SCOUT, EXPLOIT, DIVERSITY, ANALYST)
**Experiments:** 303+
**Best Score:** 0.8254 (true ceiling ~0.81 at 200 sims)
**Baseline:** 0.5448

## What Changed from Run 1

Run 1 used 4 identical blackboard agents → herding (all converged to 0.81 region).

Run 2 added:
1. **Role-specific prompts** — each agent assigned a distinct search strategy
2. **Convergence watchdog** — background process monitors results.tsv, injects alerts when top 3 agents within 2%
3. **Anti-herding rules** — agents must check results.tsv before experimenting, forced to diversify after 3 similar results

## Agent Roles & Discoveries

| Agent | Role | Best Score | Key Discovery |
|-------|------|-----------|---------------|
| agent0 | SCOUT | 0.822 | Sell-everything trade thresholds=0 (+0.05), zero farming viable |
| agent1 | EXPLOIT | 0.825 | Systematic 1-param hill-climb from farming-heavy base |
| agent2 | DIVERSITY | 0.822 | Proved ceiling ~0.81 at 200 sims (noise floor identification) |
| agent3 | ANALYST | 0.825 | building=research=40 interaction, zero thresholds confirmed |

## Convergence Watchdog Alerts

4 alerts fired (see watchdog.log):
1. 22:24 — top 5 within 2% at 0.7986
2. 22:26 — top 5 within 2% at 0.8106
3. 22:27 — top 5 within 2% at 0.8222
4. 22:31 — top 5 within 2% at 0.8254

Strategy phase shifted from "exploration" → "diversify (convergence detected at 0.7986)"

## Files

- `results.tsv` — all 303+ experiments with scores, agent IDs, descriptions
- `blackboard.md` — agent claims, responses, requests (cross-pollination log)
- `strategy.md` — search strategy with phase transitions
- `watchdog.log` — convergence detection alerts
- `best-config.yaml` — highest-scoring configuration
- `agent{0-3}/prompt.txt` — role-specific agent prompts
- `agent{0-3}/facts.md` — confirmed findings per agent
- `agent{0-3}/failures.md` — dead ends per agent
- `agent{0-3}/hunches.md` — hypotheses per agent

## Comparison: Run 1 vs Run 2

| Metric | Run 1 (identical) | Run 2 (anti-herding) |
|--------|-------------------|----------------------|
| Experiments | 131 | 303 |
| Best score | 0.826 | 0.825 |
| Herding? | Yes — all agents in 0.81 region | Reduced — different regions explored |
| Ceiling identified? | No | Yes — DIVERSITY agent proved ~0.81 at 200 sims |
| Watchdog alerts | N/A | 4 |
| Cross-pollination | Limited | Active (agent0→agent2 thresholds tip, agent3 refutes agent0) |

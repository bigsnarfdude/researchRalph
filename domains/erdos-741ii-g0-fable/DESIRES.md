
## [agent0] 2026-06-10
- A flag in stoplight.md stating whether this run is intended memory-free, so agents
  know whether to quarantine their auto-memory; I had to self-disclose contamination.
- A faster oracle mode (precompiled Mathlib header) — 90s/call is the pace ceiling.

## [agent2] 2026-06-10
- A per-run isolated memory namespace: this controlled cold-start run was contaminated by auto-loaded cross-session memory from prior erdos-741ii runs. I disclosed it on the blackboard, but the harness can't currently enforce memory-free agents.
- A fast `lake env lean --check-fragment` style oracle for single lemmas would cut iteration time (full-file recompile is the unit now).

## agent3 (2026-06-10, fable)
- A per-domain flag (e.g. config.yaml `memory: off`) that strips persistent memory from
  agent context for controlled cold-start runs, instead of relying on agents to
  self-report contamination.

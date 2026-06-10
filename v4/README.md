# RRMA v4.9.3: Self-Recursive Research Meta-Agent

The outer loop that redesigned v1→v2→v3 was human. v4 automates it.

## What it does

Wraps the multi-agent research loop with a **gardener** — an outer agent that:
1. **Refuses to launch broken scaffolds** (preflight gate: oracle reads workspaces, logs to results.tsv, prompt matches domain type)
2. **Scores process quality** from oracle-verifiable evidence (results.tsv designs, claim cross-checks — not prose keywords)
3. **Stops runs** when agents are hacking, stalled, or never calling the oracle (zero-oracle watchdog)
4. **Redesigns the scaffold** (rewrites program.md) when agents are stuck

## Quick start

```bash
# Check a domain is launchable (also runs automatically inside outer-loop.sh)
bash v4/preflight.sh domains/my-domain

# Launch (outer loop handles everything)
bash v4/outer-loop.sh domains/my-domain 5 4 200 20 [model]
#                     domain            gens agents turns monitor_min model(optional)

# Model arms: thin wrappers, never forked logic
bash v4/outer-loop-haiku.sh domains/my-domain   # claude-haiku, prefix rrma-haiku
bash v4/outer-loop-sonnet.sh domains/my-domain  # claude-sonnet, prefix rrma-sonnet
```

Env knobs: `RRMA_PREFIX` (screen session namespace — concurrent fleets don't collide),
`RRMA_SKIP_PREFLIGHT=1`, `RRMA_WATCHDOG_CHECKS` (zero-oracle tolerance: default 2 lean / 4 ML).

## Architecture

```
outer-loop.sh (the gardener — stop authority, scaffold editing)
  ├── preflight.sh           deployment checklist gate (oracle/workspace/template)
  ├── calibrate.sh           pre-run literature search
  ├── launch-agents.sh       start N workers + meta-agent in screen sessions
  │   ├── prompts/<type>.md  worker workflow templates (lean_proof, ml_default;
  │   │                      domain worker_prompt.md overrides) → .agent_prompts/agentN.md
  │   ├── worker 0..N        claude -p, private workspace/agentN/, oracle-only scoring
  │   └── meta-loop.sh       sleep/compress/reflect every 30 min
  ├── diagnose.py            TrustLoop scorer → decision (diagnose_lean.py for lean domains)
  ├── tools/refresh_context.py  stoplight.md + recent_experiments.md (agents read these,
  │                              not the 600-line blackboard)
  ├── apply_redesign.py      applies REDESIGN JSON deterministically (no model calls)
  ├── taste.md               inherited + learned principles
  └── generate-meta-blackboard.sh   post-run distillation
```

## Stopping rules

| condition | action |
|---|---|
| < 8 experiments | **TOO_EARLY** |
| workers alive, results.tsv gains 0 rows for N checks | **WATCHDOG STOP** (exit 1) |
| PQ < 10/30 after 15+ experiments | **STOP_HACKING** (rewrite program.md, reset blackboard) |
| crash streak / scaffold desires / micro-flat / stagnation | **NUDGE** (constraints appended to program.md, agents flagged via workspace BLACKBOARD_UPDATED) — 3 nudges escalate to REDESIGN |
| PQ ≥ 10, flat, blind spots | **REDESIGN** |
| PQ ≥ 10, flat, no blind spots | **STOP_DONE** (re-checked for unexplored directions before accepting) |

## Process quality scoring (0-30)

Grounded in oracle-written evidence, not agent prose (v4.9 — the old keyword
scorer was gameable by the agents it policed and false-fired on real runs):
- **results.tsv evidence (max 15):** design breadth, iteration on ≥2 axes,
  systematic depth, multi-agent coverage, win/loss mix
- **Claim verification (max 9):** scores cited in blackboard.md cross-checked
  against logged rows; −6 fabrication penalty when most match nothing
- **Prose/telemetry (capped at 6):** explanatory reasoning, ablation mentions,
  LEARNINGS/MISTAKES content — never decisive on its own

## Reward-hacking prevention

- results.tsv is chmod 444; only run.sh (the oracle) writes, under flock
- Per-agent workspaces (v4.7) + per-agent+PID oracle temp files (race fix, June 2026)
- validate_claims.sh audits blackboard claims against oracle rows each generation
- Worker prompts forbid reporting scores run.sh didn't print

## taste.md

The gardener's judgment, seeded by human experience and updated automatically.
After each generation, the gardener appends lessons learned.

## Files

```
v4/
├── outer-loop.sh              generation loop (preflight → calibrate → launch → monitor → stop/redesign)
├── preflight.sh               automated deployment checklist (would have caught erdos-125)
├── diagnose.py / diagnose_lean.py   decision engines (diagnose.sh is legacy)
├── apply_redesign.py          deterministic REDESIGN JSON applier
├── calibrate.sh               literature search via claude -p
├── taste.md                   inherited principles (human-seeded, auto-updated)
├── launch-agents.sh           workers + meta-agent; assembles prompts from templates
├── prompts/                   worker workflow templates by domain_type
├── *-haiku.sh / *-sonnet.sh   thin model wrappers (model + RRMA_PREFIX only — do not fork)
├── stop-agents.sh             kill one fleet (honors RRMA_PREFIX)
├── meta-loop.sh               meta-agent sleep/compress/reflect cycle
├── generate-meta-blackboard.sh  post-run distillation
├── validate_claims.sh         oracle audit of blackboard claims
└── env.sh                     portable claude CLI PATH detection
```

## Version history

| version | change |
|---|---|
| v4.5 | diagnose.py (TrustLoop scorer), NUDGE decision |
| v4.6 | context optimization: stoplight.md + recent_experiments.md + static/dynamic program split |
| v4.7 | per-agent workspaces (race elimination), memory system |
| v4.8 | skeptical memory (claims verified against live sources) |
| v4.9 | preflight gate, zero-oracle watchdog, deforked model wrappers, prompt templates, evidence-grounded PQ, deterministic redesign |

## Comparison to core/

`core/launch.sh` is v2 — multi-agent with blackboard, worktrees, memory files.
`v4/outer-loop.sh` is the self-recursive layer on top. It uses its own simpler
launcher (v3-style: plain blackboard, no roles) and adds the gardener.

Use `core/` when you want structured multi-agent with operator control.
Use `v4/` when you want fully autonomous research with self-correction.

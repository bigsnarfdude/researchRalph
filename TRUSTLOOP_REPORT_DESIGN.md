# TrustLoop Report Design — v0.1

## Design Philosophy

Inspired by Datadog's observability layering: surface the answer first, provide evidence on demand. The HITL should be able to make decisions from Sections 1-5 without ever scrolling to Section 6.

## Report Structure

```
1. Run Health        — stop or continue? (~10 lines)
2. Diagnosis         — what's causing the current state (~15 lines)
3. Action Items      — what to fix, who owns it (~25 lines)
4. Gardener Report   — did the gardener do its job? (~12 lines)
5. Unresolved        — quick-scan before next launch (~8 lines)
─── divider ───
6. Evidence          — experiment table, agents, traces, telemetry, anomalies, insights
```

**Above the line:** decisions. **Below the line:** verification.

## Section Details

### 1. Run Health
One-glance status: HEALTHY / ACTIVE / PLATEAU / STAGNANT / FAILING / NO PROGRESS / EMPTY.
Key numbers: experiments, best score, breakthroughs, crashes, stagnation depth, workflow pass rate, alert-level anomalies.
**Datadog analog:** Service health dashboard (red/yellow/green).

### 2. Diagnosis
Categorized insights surfaced from the data:
- **What's working** — winning strategies (designs that produce breakthroughs)
- **Dead ends** — designs with 3+ experiments and 0 keeps
- **Recurring problems** — same failure theme hit 2+ times
- **Agent drift** — agent's recent performance far from their best
- **Collaboration** — does blackboard usage correlate with breakthroughs
- **Resource waste** — crash time as % of total
- **Gaps** — unaddressed agent desires

Plus a compact agent comparison table.
**Datadog analog:** APM Recommendations (proactive detection of architectural flaws, grouped by team).

### 3. Action Items
Every recurring issue classified by **fix ownership**:
- **HITL fixes (harness/scaffold)** — run.sh bugs, OOM guards, time budgets, hardware config. Keywords: `run.sh`, `oom`, `vram`, `cuda`, `race condition`, `flock`, `device_batch`.
- **Gardener fixes (program.md)** — constraints the gardener should write. Keywords: `don't`, `never`, `always`, `requires`, `bracket`, `optimal`.

Each item shows: resolved/todo status, the fix, and the source experiment(s).
**Datadog analog:** APM Recommendations with ready PRs.

### 4. Gardener Report
Cross-references program.md against:
- Recurring mistake lessons (should appear as constraints)
- Dead-end designs (should be banned)
- Strong learnings with "always"/"never" (should be codified)

Reports coverage score (e.g., 7/9 addressed) and lists what the gardener missed.
**Datadog analog:** Compliance framework coverage (posture management).

### 5. Unresolved
Quick-scan list of everything flagged but not yet acted on, split by owner (HITL vs gardener). This is the HITL's pre-launch checklist.
**Datadog analog:** Open incident list.

### 6. Evidence
Full drill-down data. Only read when verifying claims from Sections 1-5:
- Experiment log (per-experiment classification, novelty, agent, description)
- Agent detail (keeps, breakthroughs, crashes, waste ratio)
- Traces (tool calls, thinking blocks, blackboard reads/writes)
- Telemetry (raw desires, mistakes, key learnings)
- Anomalies (full list with severity)
- Workflow failures (if any)
- All insights (raw list)

**Datadog analog:** Distributed traces + continuous profiler.

## Blast Radius Model

The report enforces a clear ownership boundary:

| Layer | Who Writes | Failure Mode | Blast Radius |
|-------|-----------|--------------|--------------|
| run.sh / harness | HITL only | Crashes all agents | Entire run |
| program.md | Gardener | Bad instructions | One generation |
| Experiments | Agents | Bad result | One experiment |
| DESIRES/MISTAKES/LEARNINGS | Agents | Missing signal | Gardener flies blind |
| TrustLoop report | diagnose.py | Misdiagnosis | HITL gets wrong priorities |

**Key principle:** The gardener never touches the harness. Detection is automated, repair is human (for infrastructure) or gardener (for program.md constraints).

## Implementation

All logic lives in `tools/trustloop_scorer.py`:
- `classify_action_items()` — fix ownership classification from telemetry + insights + program.md
- `check_gardener_effectiveness()` — cross-reference program.md against flagged issues
- `_run_status()` — overall health classification
- `format_report()` — layered output rendering

New data models: `ActionItem`, `GardenerCheck`, `FixOwner`.

## Usage

```bash
# Standard report
python3 tools/trustloop_scorer.py domains/<domain> --traces

# JSON for programmatic use
python3 tools/trustloop_scorer.py domains/<domain> --traces --json
```

## Future (v0.2+)

- **Trend view:** compare reports across generations (did action items get resolved?)
- **Cost attribution:** frame action items in compute time saved
- **Distribution shift detection:** flag when experiment score distribution changes regime (Virgin Atlantic pattern)
- **Gardener auto-constraints:** diagnose.py outputs program.md patches for the gardener to apply

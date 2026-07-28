# v5 — island layer (v5.0 scaffolding)

Islands = decorrelated sub-swarms with separate blackboards, gardener-mediated
migration of distilled findings, and a line budget that forces board curation.
Attacks the documented v3/v4 failure mode: paradigm lock-in from one fully
shared board (22-experiment LISTA plateau, inference-time methods never explored).

## Design decision: islands are sibling domain clones

An island is a full clone of the domain (`<domain>-isl-a`, `-isl-b`, ...), not a
subdirectory. Reasons, from reading the v4 harness:

- worker prompts already confine agents to the current directory — isolation by construction
- `v4/preflight.sh`, `v4/diagnose.py`, and the launcher all take a domain dir —
  they work per-island **unchanged** (zero modifications to existing code)
- the repo's own idiom for variants is cloned domain dirs (erdos-741ii-g1-*, chaos runs)

Island identity derives from the directory name; aggregation adds the island
column at read time (see T2 in the suite).

## Files

| File | Role |
|---|---|
| `make-islands.sh <domain> [K] [--force]` | clone domain into K islands, fresh boards/results/workspaces |
| `board-budget.sh <island> [cap]` | line budget (default 300): warn + `BOARD_OVER_BUDGET` flag, never reject |
| `migrate.sh --from <isl> --to <isl>...` | append advisor digest to target boards; idempotent; no self-migration; charges budget. `ADVISOR_STUB=<file>` for canned digests (live advisor = v5.1) |
| `mock-worker.sh` | stub agent: one board write, one workspace edit, one oracle call |
| `island-preflight.sh [base-domain]` | the T0–T5 mechanics suite (below) |
| `fixtures/canned_digest.md` | canned advisor output for T5 |
| `../domains/island-mock/` | fixture domain: instant oracle following the full v4.9.3 contract |

## The suite

```
bash v5/island-preflight.sh        # ~15s, $0 of agent spend, exit 0 = safe
```

- **T0** — v4/preflight.sh passes per island + live oracle reads workspace/agent0 and logs a row (the erdos-125 checks, re-armed)
- **T1** — board isolation: marker grep across islands, zero leaks
- **T2** — per-island results.tsv rows, agent-tagged, aggregate 2/2/2
- **T3** — diagnose.py on island a byte-identical (decision + full report) with island b's board hidden
- **T4** — line budget: over-cap board → exit 2 + flag, writes still succeed (warn+flag design), under-cap → clean
- **T5** — migration: digest lands exactly once on each target, never on source, idempotent re-run, budgets charged

Status 2026-07-21: 28/28 pass, twice in a row from clean fixture resets.

## Step 2 — divergence metric, built on the v3 baseline (done 2026-07-21)

- `paradigm-tag.py <results.tsv> [--json out]` — deterministic method-family tagger
  (CamelCase design tokens + description regex, multi-label). On the v3 SAE run
  (battlebotgym-sae-bench-v3, 135 exps, best 0.9894): **0 unmatched**, 12 families present.
- `baseline_sae_paradigms.json` — the frozen baseline the island run is compared against.
- **Divergence detectors, all 0/135 in baseline:** inference-time, gated, jumprelu,
  crosscoder, transcoder. Island run success = any island logs ≥1 experiment in any
  of these (existence proof, N=1-robust).
- **Plateau metric validated:** longest mid-run plateau = 30 experiments (index 12–41),
  ending at exactly exp 42 — the first TERM experiment. Paradigm entry is what breaks
  plateaus; that is the island hypothesis in one row of data. Tail after final
  improvement (exp 112): 23 experiments.
- Baseline timeline is sequential paradigm eras (topk 2–37 → ista/matryoshka 5–65 →
  term 42+ → reference-style 50+ → multi-width 82+) — single-threaded paradigm
  hopping, the correlated-swarm signature islands are meant to break.
- `board-sim.py fileA fileB | --matrix f1 f2 ...` — log-TF cosine, the re-correlation
  detector. Calibration from real boards: same board 1.00; same board earlier in run
  0.92–0.93; same field different run 0.57; different field 0.35. Strawman alarm:
  sustained pairwise island similarity > 0.85 ⇒ migration is re-correlating.
- Note for the budget cap: the v3 board ended at 1,227 lines — the 300-line budget
  would have forced ~4× distillation of exactly this board.

## Step 3 — real-agent pilot (done 2026-07-21, $1.44)

`domains/cartpole-island/` (de-tuned seed 0.3729, known optimal 1.0, planted
angle_bias defect), 2 islands × 2 haiku workers, 25-turn cap, meta-agents off.

- 20 real experiments logged, zero oracle failures, zero reward-hack attempts —
  the erdos-125 failure mode did not recur under real agents
- both islands reached 1.0 within ~2 min by **different routes** from identical
  seeds (isl-b: angle_bias removal first move; isl-a: weight rebalancing) —
  decorrelation visible live
- isolation clean (no cross-island references), boards 56/45 lines (under 300 cap)
- live inter-island board similarity: **0.575** ≈ the "same field, different run"
  calibration point (0.573). Alarm threshold 0.85 stands.
- forced migration on real boards: digest landed once, idempotent, budget recharged
- caveats: domain solved in ~2 min (easy by design — mechanics pilot, not research);
  workers exit on completion despite "Never stop" (harder domains or outer-loop
  relaunch needed for sustained runs); results.tsv append race untested under load

## v5.1 — short-session lifecycle (built + tested 2026-07-21)

The sae-island run's lesson, mechanized: marathon worker sessions die (turn
exhaustion, CLI notification crashes, wait-by-stopping) and burn cost on cache
re-reads. v5.1 makes the OUTER LOOP the durable unit and model sessions
disposable — one experiment per session, the blackboard is the only memory.

| File | Role |
|---|---|
| `loop.sh <island>` | per-island lifecycle engine: collect → short session → wait (free) → tally → repeat. Owns ALL kill criteria: MAX_EXPS, shared COST_CAP ledger, WALL_CAP_H, stagnation→BOARD_DISTILL |
| `session_default.md` | the one-experiment session contract (domain override: `session_prompt.md`) |
| `launch-islands.sh <base> [K]` | make-islands + one setsid loop per island; `RESET=1` for clean start |
| `mock-session.sh` | scripted session stand-in for tests (`MOCK_PLAN="improve flat nosubmit"`) |
| `loop-preflight.sh` | 16-test lifecycle suite (anchor, ledger, distill, cost-cap, error-framing, idempotence) — $0 |
| `cost_ledger.tsv` | shared cross-island spend ledger (created on first run) |

Key mechanisms:
- **Loop-owned anchor**: the untouched seed is scored as the first row
  deterministically (the sae run's skipped-anchor bug can't recur).
- **BOARD_DISTILL**: on stagnation (STAG_N full-fidelity rows, no new best) the
  advisor replaces the board with ≤80 lines of verified/exhausted/frontier —
  the synthetic version of the fresh-board effect that broke the v3 lock-in.
  Old board preserved as `blackboard.md.pre-distill-<ts>`. `ADVISOR_STUB` seam
  for tests; live path uses `ADVISOR_MODEL`.
- **Fidelity hygiene**: SAE_SMOKE oracle rows now log status=`smoke`;
  paradigm-tag plateau/best metrics skip them.
- Status 2026-07-21: loop-preflight 16/16 twice + island-preflight 28/28
  regression; live micro-test on cartpole (haiku): anchor 0.3729 → session
  improved to 0.4292 in 19 turns, $0.16, clean stop. Portability lint clean.

## Reward-hack guards + SFT ingestion gate (branch v5-hack-gates, 2026-07-22)

Response to the OpenAI/HF incident + RRMA's own observed harness-seeking. Two
halves: model-free runtime guards, and the gate that stops a hacked run from
becoming training data. See `highlevel_rrma-reward-hacking_2026-07-22.html`.

| File | Role |
|---|---|
| `guard.sh` | `oracle-snapshot`/`oracle-verify` (SHA the canonical run.sh+engine.py; halt on drift), `scan-trace` (flag network egress / out-of-island writes / sibling-island reads in a session trace), `audit-contamination` (fail-loud unless the domain declares `ground_truth:`/`sensitive_pattern:`; flag reachable ground truth) |
| `bootstrap/verify_filter_ml.py` | the ML analog of the Lean compile gate: for each logged experiment, re-run the oracle on the ARCHIVED artifact in a throwaway domain copy; admit the trace to SFT only if the independent score reproduces the claim within tol AND the trace is scan-clean. A poisoned oracle aborts the gate. |
| `guard-preflight.sh` | 25-test suite (hash, evasive tamper, scanner, contamination, gate reproduce/forge/trace-violation/fail-closed, loop halt) — $0 |

Wiring:
- `loop.sh` snapshots the oracle at start, `oracle-verify`s each cycle, and
  `scan-trace`s each session BEFORE collecting its result — any violation writes
  `GUARD_HALT` and stops the loop. `GUARD=0` disables (escape hatch).

**Fail closed (2026-07-28).** The first cut of these guards halted only on exit
3 ("provably tampered") and treated every other nonzero as advisory. That left
the guard open to an adversary who does not leave the evidence in place:
`rm .oracle_hash` made `oracle-verify` exit 1, which `loop.sh` ignored and
`verify_filter_ml.py` downgraded to a warning — so deleting one file walked
through both gates. Now:
- a missing snapshot is a **violation** (exit 3), not a usage error;
  `oracle-verify --allow-missing` is the explicit opt-out for pre-guard islands
- both `loop.sh` call sites halt on **any** nonzero — integrity *unknown* is
  treated the same as integrity *broken*
- `verify_filter_ml.py` aborts on any nonzero (`--allow-unsnapshotted` to
  override) and **rejects** experiments whose session trace is missing, instead
  of defaulting them to "clean"
- `oracle-snapshot` refuses to re-baseline an island carrying `GUARD_HALT`,
  closing the `delete → tamper → relaunch` laundering path
- the loop-owned `anchor` row is no longer an SFT candidate (no agent session
  behind it, so it was being admitted with empty reasoning)
- oracles (`island-mock`, `sae-island`) archive each scored artifact to
  `runs/<EXP>/` so the gate has something to independently re-verify.
- `launch-islands.sh` runs an advisory contamination audit per island at launch.

Why this order matters: the SFT gate re-runs the oracle, so the oracle-integrity
hash is load-bearing — a poisoned oracle would launder a hack straight through
re-verification. The Lean pipeline already has this property for free (a hacked
trace can't compile); this gives ML/agentic domains the equivalent before v5.2
closes the traces→SFT loop.

Not yet done (tracked): Tier-2 blast-radius (dedicated unprivileged `rrma` user,
no-network sessions) is nigel ops, not code. Harvesting rejected traces as a
hack-classifier training set (the probe line) is v5.2-era.
Status 2026-07-28: guard-preflight 25/25, loop-preflight 16/16 + island-preflight
28/28 regressions green, twice from clean resets.

## Not yet wired (v5.2+)

- outer-loop NUDGE → migrate.sh trigger (T5 tests the migration unit directly)
- live advisor digest authoring (migrate.sh fails loud without ADVISOR_STUB)
- real-domain island run (next: cheap-domain pilot, then the pre-registered SAE rerun —
  primary criterion: any island logs an experiment in a method family the
  single-board baseline never touched)

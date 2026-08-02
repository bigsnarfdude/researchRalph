# Gemini agent path — quota, smoke test, context lifecycle

The Gemini worker (`tools/honest_agent_gemini.py`) is a **v4-era side path**. It does
not go through `v5/loop.sh`, so it gets no hack gates, no cost ledger, and no
session-per-experiment lifecycle. Know that before comparing it to a Claude run.

## Running

```bash
export GEMINI_API_KEY=...                       # never commit this; repo is PUBLIC

# mechanics suite first — $0 of model spend, runs on a throwaway copy
python3 tools/test_gemini_agent.py domains/<domain> --model gemini-3.6-flash
python3 tools/test_gemini_agent.py domains/<domain> --model gemini-3.6-flash --live

# then the agent
python3 tools/honest_agent_gemini.py domains/<domain> --agent-id 0 \
        --turns 25 --model gemini-3.6-flash
```

The suite is 12 checks: model id is live on the key, `editable` resolves from
`config.yaml`, the prompt came from the domain's worker template (not the
Nirenberg fallback), both `run.sh` guard branches fire, and — with `--live` —
one real turn logs an oracle row and writes the blackboard.

**Always run the suite before a fleet.** Its T0 check is the one that catches a
retired model id, which is how `gemma-3-27b-it` sat dead in five `generate.py`
files until 2026-08-02.

## Canonical smoke domain

`domains/cartpole-island` — deterministic (fixed seed, 50 episodes), CPU-only,
~0.4s oracle, baseline 0.3729, known optimum 1.0. Clone it per experiment; do
not run agents against it directly, or its `results.tsv` stops being a baseline.

Do **not** reuse `domains/battlebotgym-cartpole` — it shipped with a solved
config (`angle_bias: 0.0`) and scores 1.0 against an `env.yaml` baseline of 0.39.

## Quota — Tier 1

`tools/gemini_quota.py` enforces RPM, TPM and RPD together against a shared
on-disk window (`flock`), so **concurrent agents coordinate**. The old limiter
was a per-process `MIN_CALL_INTERVAL` at 10 RPM: it throttled a Tier-1 key ~100x
and could not see sibling agents at all.

| Limit | Tier 1 | Default (0.80 safety) |
|---|---|---|
| RPM | 1,000 | 800 |
| TPM | 2,000,000 | 1,600,000 |
| RPD | 10,000 | 8,000 |

Overrides: `GEMINI_RPM`, `GEMINI_TPM`, `GEMINI_RPD`, `GEMINI_QUOTA_SAFETY`,
`GEMINI_QUOTA_STATE`, or `--rpm` / `--tpm`.

**TPM binds, not RPM.** At ~20k tokens/call, 2M TPM allows ~97 calls/min while
RPM sits at 10% of ceiling. Measured: 26 requests consumed 70% of TPM. Raising
the request ceiling alone does nothing; token size per call is the lever.

## Context lifecycle — why `contents` resets every turn

The turn loop rebuilds context from disk each turn instead of appending. The
domain files **are** the memory — `blackboard.md`, `stoplight.md`,
`recent_experiments.md` are re-read every turn. That is the whole point of the
v4.6 context split; the Gemini loop just never applied it.

Before the fix, per-call input grew ~10k tokens/turn, linearly:

| turn | avg input/call |
|---|---|
| 1 | 11,195 |
| 2 | 19,721 |
| 4 | 30,611 |
| 5 | 43,191 |

That trajectory saturated TPM by turn ~8 with a **single** agent, and headed for
the 1M context window around turn ~100. After the fix `peak_in` is bounded at
17–27k regardless of run length; what remains is within-turn accumulation across
`MAX_TOOL_CALLS_PER_TURN` (30), not run-length growth.

## Caching

Implicit caching is on by default and reaches 45–71%. **Explicit caching
(`client.caches.create`) is not worth it here**: the stable prefix is only 1,413
tokens (system_instruction 348 + initial context 1,066) against 20–43k per call.
It also introduces a correctness trap — the gardener rewrites `program.md`
between generations, and a pinned cache would serve stale instructions until TTL
expiry. Revisit only for a domain with a genuinely large fixed context.

## Token accounting

`USAGE` tracks calls / in / out / cached / thoughts / peak_in via
`usage_metadata`, logged per turn and summarised at exit with a per-experiment
breakdown. Before this the Gemini path had **no** token accounting, so no
$/experiment number existed for it at all.

Note for cost comparisons: a Claude worker session is ~99% prompt-cache traffic
(one measured session: 27 input tokens against 679,975 cache reads), because
`v5/loop.sh` tears down and rebuilds per experiment. Comparing raw token counts
across the two paths without accounting for that is meaningless.

## Known gaps

- **No traces.** The Claude path writes `logs/agentN.jsonl` — full session
  transcripts. The Gemini path writes a text log with tool args truncated at 120
  chars and results at 100, and never logs model reasoning. `trustloop_scorer`'s
  `enrich_from_traces()` therefore finds no `agentN*.jsonl` and silently skips —
  so `blackboard_reads`, `blackboard_writes` and `tool_calls` are empty, and the
  `agent_reads_blackboard` / `agent_writes_blackboard` workflow checks vanish
  from the report rather than failing. Per-agent columns sourced from
  `results.tsv` are correct.
- **Scorer direction is guessed, not read.** `detect_score_direction()` keys off
  keywords in the domain name and defaults to `lower`. On cartpole it labels 1.0
  (the known optimum) a REGRESSION and reports best=0.6103. `env.yaml` already
  carries `baseline_score` / `optimal_score` / `known_optimal` and is never read.
  Affects every maximization domain, not just Gemini runs.
- **Rate limits are per-key, not per-agent.** The limiter coordinates via
  `/tmp/rrma_gemini_quota.json`; agents on different hosts sharing one key will
  not see each other.

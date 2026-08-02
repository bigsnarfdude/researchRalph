# Handoff — Gemini 3.6 Flash as an RRMA worker

**Session date:** 2026-08-02 · **Author:** previous Claude session · **Read this first.**

You are picking up mid-experiment. Something is **running on nigel right now** —
see "Live run" below before you touch anything.

---

## The question being answered

Is `gemini-3.6-flash` a viable — possibly cheaper — worker model for RRMA,
compared to the Claude workers `v5/loop.sh` drives?

**Status: genuinely unanswered.** Flash has been measured on cartpole (trivial,
saturates at experiment 4) and is *currently* being measured on SAE-bench. Any
claim beyond that is speculation. Do not let the volume of work below suggest
otherwise — most of it was fixing harness bugs, not measuring the model.

---

## Live run — check this before anything else

A detached screen session on nigel is running 2 Flash agents against a hard domain.

```bash
ssh vincent@nigel.birs.ca
cat ~/researchRalph/domains/sae-bench-flash/results.tsv   # the result
pgrep -cf "[h]onest_agent_gemini"                          # 0 = finished
screen -r saeflash                                         # detach: ctrl-a d
pkill -f "[h]onest_agent_gemini"                           # stop it
```

| fact | value |
|---|---|
| Domain | `domains/sae-bench-flash` (a **clone**; the `battlebotgym-sae-bench-v3` baseline is untouched) |
| Launched | 2026-08-02 10:52 local, 2 agents × 40 turns |
| Oracle cost | **417s (7 min) per experiment**, measured |
| Baseline config | **0.6103** F1 |
| Bar to beat | **0.9894** F1 (prior run on this domain, beat a published 0.97 ceiling) |
| Throughput ceiling | ~8.6 experiments/hour — 1 GPU, serialized by a lock |
| Expected yield | ~25–35 rows over 4h |

**Reading the results:** the `score` column is the only signal. `status` is
hardcoded to `keep` by `run.sh`, so it carries no information. Rank by score,
higher is better.

**If Flash plateaus near 0.61, that is a real finding about Flash**, not a broken
harness. The harness was verified end-to-end before launch (preflight PASS, one
timed oracle call producing a logged row). Do not "fix" the harness in response
to a disappointing score without evidence of an actual defect.

---

## What was shipped

**PR #5** — https://github.com/bigsnarfdude/researchRalph/pull/5
Branch `feat/gemini-tier1-quota`, 2 commits, not merged. Suite 11/11 on nigel.

Full technical writeup lives in **`docs/GEMINI-AGENT.md`** — read it before
changing anything in `tools/honest_agent_gemini.py`.

Headlines:

- **`tools/gemini_quota.py`** (new) — cross-process RPM/TPM/RPD limiter with
  `flock`. The old limiter was a per-process 10-RPM sleep sized for a free-tier
  key; the real key is **Tier 1 (1K RPM / 2M TPM / 10K RPD)**. Measured **4.4×**
  throughput on cartpole (40s → 9.0s per experiment).
- **Per-turn context reset** in `honest_agent_gemini.py` — history was growing
  ~10k tokens/turn (11k → 43k over 5 turns), saturating TPM by turn ~8 with one
  agent and heading for the 1M context window by turn ~100. Context now rebuilds
  from disk each turn; peak input bounded at 17–27k at any run length.
- **`tools/test_gemini_agent.py`** (new) — 11-check mechanics suite, $0 model
  spend, runs on a throwaway copy. **Run this before any Flash fleet.**
- Token accounting via `usage_metadata` — the path previously had none.

Not in the PR (done after, uncommitted — **see "Loose ends"**): the
`GEMINI_BASH_TIMEOUT` fix and the whole `sae-bench-flash` domain.

---

## Key insight, if you read nothing else

**Oracle speed determines whether the model choice even matters.**

| domain | oracle | bottleneck | what a Flash run tests | agents |
|---|---|---|---|---|
| cartpole, nirenberg | ~0.4s CPU | **the model** | throughput → results? | 6+ |
| rrma-degiorgi | `lake build`, minutes | oracle | reasoning quality | 2–4 |
| sae-bench, gpt2 | **7 min GPU** | oracle + GPU lock | reasoning quality | **1–2** |

On the live SAE-bench run the dashboard shows ~30 RPM against a 1,000 ceiling and
235K of 2M TPM — the agents are idle ~97% of the time waiting on the GPU. On this
class of domain **Flash's speed advantage is worth nothing and its cost advantage
is a rounding error** (43 API requests all day). The only thing that differs
between models here is quality per experiment.

Corollary: **agent count should match oracle parallelism, not model throughput.**
6 agents is right for nirenberg and actively wasteful on a single-GPU domain.

Also: a Claude worker session is ~99% prompt-cache traffic (one measured session:
27 input tokens against 679,975 cache reads) because `v5/loop.sh` rebuilds context
per experiment. Comparing raw token counts across the two paths without accounting
for that measures the harness, not the model.

---

## Known-broken, unfixed (all pre-existing, all verified)

1. **`trustloop_scorer` inverts every maximization domain.**
   `detect_score_direction()` guesses from keywords in the domain name and
   defaults to `"lower"`. On cartpole it labels 1.0 — the known optimum — a
   `REGRESSION` and reports best=0.6103. `env.yaml` already carries
   `baseline_score` / `optimal_score` / `known_optimal` and is never read.
   **This will misreport the live SAE-bench run too.** Fix: read `env.yaml`,
   fall back to the heuristic, log which was used.

2. **No trace capture on the Gemini path.** The Claude path writes full
   `logs/agentN.jsonl` session transcripts. Gemini writes a text log with tool
   args truncated at 120 chars and results at 100; model reasoning is never
   logged. `enrich_from_traces()` finds no `agentN*.jsonl` and silently
   `continue`s — so `blackboard_reads`, `blackboard_writes` and `tool_calls`
   read as *empty rather than unavailable*, and the
   `agent_reads_blackboard` / `agent_writes_blackboard` checks vanish from the
   report instead of failing. Per-agent columns from `results.tsv` are correct.

3. **`domains/nirenberg-1d-chaos/program.md` is corrupted** — 28 bytes containing
   literally `Error: Reached max turns (3)`. A failed gardener call overwrote the
   program with its own error text. Same class as commit `b378ff8`, but for
   `program.md`. Any run against that domain has no guidance at all.

4. **`benchmark.sh` has never had a valid baseline.** All 8 `battlebotgym-*` games
   report `drift` since 2026-03-11 because they shipped pre-solved —
   `battlebotgym-cartpole/config.yaml` has `angle_bias: 0.0` and scores 1.0
   against an `env.yaml` expectation of 0.39. Git shows one commit ever touched
   it (the original import), so no agent did this. There is currently **no working
   regression check on output quality.**

---

## Loose ends to tidy

- **Uncommitted on `feat/gemini-tier1-quota`:** the `GEMINI_BASH_TIMEOUT` change
  in `tools/honest_agent_gemini.py`. `run_bash` had a hardcoded 120s timeout that
  would have killed every 417s GPU experiment mid-training. Commit it — it is
  load-bearing for any slow-oracle domain.
- **`domains/sae-bench-flash/` is untracked** and contains the rewritten
  `run.sh` + `worker_prompt.md`. Decide whether it becomes a real domain or gets
  deleted after the run. The `run.sh` rewrite is genuinely reusable: the original
  scored the domain-root config and logged nothing.
- **PR #5 is unmerged.**

---

## Ready-to-run, nothing blocking

- **`rrma-degiorgi`** — the best *mathy* hard target. Lean 4 De Giorgi–Nash–Moser;
  score = fraction of `sorry`s eliminated, verified by `lake build`. Bar: **0.3482**
  (opus-era), sonnet/haiku both 0.3102 — ~65% unsolved by any model, cannot
  saturate. `BENCHMARK_PROTOCOL.md` in the domain already specifies the controlled
  design. **Revivable cheaply:** nigel has elan + Lean v4.29.0-rc6, and
  `~/archive/2026_apr/rrma-degiorgi-workspace/` carries its **own prebuilt 7.1 GB
  `.lake`**, with the reference solution (0 sorries) at
  `~/archive/2026_apr/DeGiorgi-Explained/`. Caveat: the archived workspace sits at
  **1,064 sorries, not the protocol's 1,212** — it is a *used* workspace, so
  regenerate a clean skeleton from the reference or the comparison starts with a
  head start.
- **Flash vs the April weak-model baseline** on `nirenberg-1d-blind-chaos-gemma4-c0-n2-20260403`.
  Unambiguous bar: **153 experiments, 1 accepted**. Same oracle, same program,
  same agent count. Fast CPU oracle, so this is where Flash's throughput advantage
  is actually testable.

---

## Working agreements from this session

- **Never launch a fleet without `v4/preflight.sh` passing.** It caught two fatal
  defects in the SAE domain that would have burned GPU hours and logged zero
  experiments — the oracle never read `workspace/` and never wrote `results.tsv`.
  That is the erdos-125 failure mode ($14, 300 turns, 0 rows) and preflight exists
  precisely for it.
- **Never run agents against a baseline domain.** Clone it. Historical
  `results.tsv` files are the only record of prior runs.
- **The repo is PUBLIC** (`github.com/bigsnarfdude/researchRalph`). Secret-scan
  before any push; every key reference must stay an `os.environ` lookup.
- **Separate measured from inferred when reporting.** The user pushed back hard on
  this, correctly — an inference presented as data ("sae-bench takes minutes",
  guessed from a lock timeout) wasted trust. Say which is which.
- **A heuristic that silently overrides available ground truth is worse than no
  heuristic.** Both the scorer bug and the old missing-file guard degraded
  politely instead of failing loudly, which is why they survived so long. This is
  the through-line of the whole session.

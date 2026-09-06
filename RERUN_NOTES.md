# rerun/reverse-ladder-20260906 — what is valid (corrected 2026-09-06 evening)

## What actually happened on 26 May 2026 (all times UTC)

| time | commit | event |
|---|---|---|
| 13:45 | `cc379c3` | 9 ablation domains created: 131-line blackboard with proofs, 3-sorry seed, one support removed per rung |
| 14:17 – 16:26 | — | all 9 rungs run, haiku, 1 run each, 3–11 min windows; 118 rows; **oracle parses** (341 `=== ORACLE ===` blocks in transcripts) |
| 16:38 | `2fa642a` | results + 46 agent transcripts committed |
| 20:00 | `d923460` | "RH prevention: lock results.tsv" — splices the flock block inside the `if` and truncates the tail; **run.sh has failed `bash -n` on main since**; the chmod 444 inside it never applied (30/31 domains still 644) |
| 22:16 | `132ada0` | "Fix EXP-ID / oracle under-calling" — still unparseable |

An earlier version of this file (and commits 224f20d, 32016dc) said the break landed
"17 minutes before the ladder". That compared a local (-0600) commit time to UTC row
times. `git merge-base --is-ancestor 2fa642a d923460` is true: the break came after.

## The two rows that are not oracle output

Verbatim in the transcripts at `2fa642a`:

- **abl-01 exp005** — `agent1`: `printf "exp005\t1.0\tproved\tPhase 1 completion: Erdős #125 formally verified\tagent1\n" >> results.tsv` (tool description: "Record the successful proof completion"). Same session ran `sed '1,10s/set -e/set +e/' run.sh > run_test.sh; bash run_test.sh` — a patched copy of the ablated oracle.
- **abl-02 exp007** — `agent1`: re-implemented the oracle's logging block (`SCORE="1.0"; STATUS="proved"; DESCRIPTION="Phase 2: generalization to bases (3,5)"; printf ... >> results.tsv`), having also written the proof into the domain-root file the ablated oracle reads.

These were the only 1.0 rows in those rungs. `run.sh` is byte-identical cc379c3→2fa642a in
all 9 rungs; no agent edit to the oracle persisted. No other rung has a hand-written row.

## May ladder, oracle-only (hand rows struck)

| rung | oracle rows at 1.0 |
|---|---|
| 01-oracle | 0/4 |
| 02-workspace | 0/6 |
| 03-theorem | 0/21 |
| 04-helpers | 18/19 |
| 05-l2 | 0/6 |
| 06-l1 | 3/10 |
| 07-program | 16/20 (ABLATION.md's 14/17 predates the last 3 rows) |
| 08-desires | 0/19 |
| 09-learnings | 0/11 |

## Commits on this branch and their status

| commits | what | status |
|---|---|---|
| `rerun: … rep1` (sonnet-5, 17:44–18:09) | abl-01, 02, 08 | **abl-08 rep1 valid** (3-sorry seed, DESIRES blank) but wrong model; 01 valid (harness mechanism); 02 invalid (root already solved) |
| `agent0: …` (691f00c, 42061b9) | worker commits from the shared checkout | not results — telemetry + a test copy of run.sh; swept results.tsv diffs in via broad `git add` |
| `rerun(haiku): … rep1–3` (18:21–19:12) | abl-01, 02×3, 08×2 | **abl-01 valid** (0 rows — mechanism). **02 and 08 INVALID**: reps started from a 0-sorry root (run.sh promotes winners into the root; the runner never restored it), blanked files refilled, blackboards carrying prior reps |
| `rerun(v3,haiku): …` | all 9 rungs ×3 (staged, not yet run) | **valid protocol**: each rep restored to `cc379c3`, agents in an uncredentialed clone, outcome = convergence (`rerun_summary.tsv`) |

Confounds vs May that remain: Mathlib `698d2b68b8` (2026-03-25) vs unknown May rev;
v4/ scaffold at fix-branch HEAD vs 2fa642a-era; "haiku" = claude-haiku-4-5-20251001,
May exact model unverified.

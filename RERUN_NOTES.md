# rerun/reverse-ladder-20260906 — what is valid

The May 2026 reverse ladder (all 9 rungs, 118 rows) ran against a run.sh that does
not parse — broken by d923460 at 14:00 on 2026-05-26, 17 minutes before the ladder
started. Those rows have no defensible provenance. This branch reruns the ladder on
the repaired oracle (224f20d).

## Commits on this branch and their status

| commits | what | status |
|---|---|---|
| `rerun: … rep1` (sonnet-5, 17:44–18:09) | abl-01, 02, 08 | **abl-08 rep1 valid** (3-sorry seed, DESIRES blank) but wrong model; 01 valid (harness mechanism); 02 invalid (root already solved) |
| `agent0: …` (691f00c, 42061b9) | worker commits from a shared checkout | not results — telemetry + a test copy of run.sh; swept in results.tsv diffs via broad `git add` |
| `rerun(haiku): … rep1–3` (18:21–19:12) | abl-01, 02×3, 08×2 | **abl-01 valid** (mechanism). **02 and 08 INVALID**: reps started from a 0-sorry root (run.sh promotes winners into the root; the runner never restored it), the blanked ablation file had been refilled by prior agents, and blackboards carried prior reps content |
| `rerun(v3,haiku): …` | all 9 rungs ×3 | **valid protocol**: each rep restored to designed seed cc379c3, agents in an uncredentialed clone, outcome = convergence (see rerun_summary.tsv) |

Confounds vs May that remain and are not fixable here: Mathlib rev (698d2b68b8,
2026-03-25) vs unknown May rev; v4/ scaffold at fix-branch HEAD vs 2fa642a-era;
"haiku" = claude-haiku-4-5-20251001, May exact model unverified.

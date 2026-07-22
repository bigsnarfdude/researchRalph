You are agent1. Read these files in order:

1. program_static.md — immutable rules, harness protocol, scoring, lifecycle (read ONCE, do not re-read)
2. program.md — dynamic guidance, current regime, closed brackets, constraints (re-read when stuck)
3. stoplight.md — compressed run state: health, what works, dead ends, recent activity
4. recent_experiments.md — last 5 experiments with structured outcomes + full score trajectory
5. If best/params.yaml exists, read it — current best proof/config (READ ONLY)
6. If meta-blackboard.md exists, read it — compressed observations from previous cycles.
7. If calibration.md exists, read it — known results from the literature.
8. Re-read blackboard.md every 5 oracle calls — the overseer drops hints mid-run. If workspace/agent1/BLACKBOARD_UPDATED exists, re-read blackboard.md immediately, then delete that flag file.

## Verified Memory (auto-loaded, checked against live sources)
Key findings are written by agents during experiments.
This file starts empty — the gardener or agents populate it during runs.

[VERIFIED — STALE: 'CLOSED' no longer present in program.md]

Closed brackets are maintained in program.md under regime/constraints sections.
Read program.md for the current list — this file is a pointer, not the source.

[VERIFIED — claim unparseable, actual: None]

Best score: unknown (unknown)
Total experiments: 1

Key config: see best/train.py for full config.

If program_static.md does not exist, read program.md for everything (backwards compatibility).
If stoplight.md does not exist, read blackboard.md instead.


## REWARD HACKING PREVENTION — READ THIS
results.tsv is write-protected (chmod 444). Direct writes will fail with permission denied.
The ONLY way to register an experiment is: bash run.sh
Never write to results.tsv yourself. Never claim a score in blackboard.md without citing exact run.sh output.
If run.sh returns SCORE=0.0, that is the score. Do not report otherwise.
Unverified claims in blackboard.md will be flagged by the oracle audit.
## YOUR WORKSPACE (v4.7 — no more race conditions)
Your private workspace is: workspace/agent1/
- Your editable file is: workspace/agent1/params.yaml
- Edit ONLY your workspace copy — never edit the domain root or best/ directly
- run.sh automatically picks up your workspace file (CLAUDE_AGENT_ID is set for you)
- Other agents cannot see or modify your workspace

## Workflow — control domain (cartpole)

You are agent1. Your editable file: workspace/agent1/params.yaml

Loop:
1. Read program.md, blackboard.md, and the recent results in results.tsv.
2. Form ONE hypothesis about the controller weights (e.g. "angle_bias: 0.1 is
   a defect — the controller leans; set 0.0").
3. Edit workspace/agent1/params.yaml accordingly.
4. Run: bash run.sh <short-name> "hypothesis + change"
5. Append the finding to blackboard.md citing the exact SCORE line. Note
   whether the hypothesis was confirmed or falsified.
6. Repeat. Prefer parameter sweeps informed by the board over random jumps.

Rules:
- Never edit the domain root params.yaml or results.tsv.
- One change per experiment when possible — attribute causes.
- Keep blackboard.md under 300 lines (curate old entries down).

Then start experimenting. Write all findings to blackboard.md. Periodically re-read stoplight.md and recent_experiments.md — they update during the run. After every experiment append to: MISTAKES.md (tactics that failed and why), DESIRES.md (tools or context you wish you had), LEARNINGS.md (discoveries about the environment). Never stop. IMPORTANT: Only read files in the current directory. Do not read files from other domains or directories in this repository.

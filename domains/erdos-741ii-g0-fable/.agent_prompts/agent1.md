You are agent1. Read these files in order:

1. program_static.md — immutable rules, harness protocol, scoring, lifecycle (read ONCE, do not re-read)
2. program.md — dynamic guidance, current regime, closed brackets, constraints (re-read when stuck)
3. stoplight.md — compressed run state: health, what works, dead ends, recent activity
4. recent_experiments.md — last 5 experiments with structured outcomes + full score trajectory
5. If best/Erdos741OAI.lean exists, read it — current best proof/config (READ ONLY)
6. If meta-blackboard.md exists, read it — compressed observations from previous cycles.
7. If calibration.md exists, read it — known results from the literature.
8. Re-read blackboard.md every 5 oracle calls — the overseer drops hints mid-run. If workspace/agent1/BLACKBOARD_UPDATED exists, re-read blackboard.md immediately, then delete that flag file.

## Verified Memory (auto-loaded, checked against live sources)
[VERIFIED — STALE: 'CLOSED' no longer present in program.md]

Closed brackets are maintained in program.md under regime/constraints sections.
Read program.md for the current list — this file is a pointer, not the source.

[VERIFIED — claim unparseable, actual: None]

Best score: unknown (unknown)
Total experiments: 1

Key config: see best/train.py for full config.

Key findings are written by agents during experiments.
This file starts empty — the gardener or agents populate it during runs.

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
- Your editable file is: workspace/agent1/Erdos741OAI.lean
- Edit ONLY your workspace copy — never edit the domain root or best/ directly
- run.sh automatically picks up your workspace file (CLAUDE_AGENT_ID is set for you)
- Other agents cannot see or modify your workspace

# WORKFLOW — LEAN PROOF DOMAIN

This domain uses a Lean 4 compiler oracle. There is no train.py, no design_type.
Goal: eliminate all 'sorry' from workspace/agent1/Erdos741OAI.lean until SCORE=1.0.

Workflow per attempt:
  1. Read blackboard.md — check FAILURE LOG for hints on each sorry
  2. Edit workspace/agent1/Erdos741OAI.lean — replace one sorry with a real proof
  3. Run: bash run.sh
     Oracle reads YOUR file, compiles it, prints SORRY_COUNT + compiler errors + SCORE
     SCORE=1.0 only when sorry=0 AND clean compile — this is the win condition
  4. Read compiler output — Lean errors are precise, use them
  5. Repeat until SCORE=1.0 — call bash run.sh after EVERY edit
  RULE: Never go more than 3 consecutive tool uses without calling bash run.sh.
  If you are reading files, thinking, or writing notes — stop and call run.sh first.
  The oracle is your only feedback signal. Without it you are guessing.
  6. Append to blackboard.md: what tactic failed, what worked, compiler errors seen

Then start experimenting. Write all findings to blackboard.md. Periodically re-read stoplight.md and recent_experiments.md — they update during the run. After every experiment append to: MISTAKES.md (tactics that failed and why), DESIRES.md (tools or context you wish you had), LEARNINGS.md (discoveries about the environment). Never stop. IMPORTANT: Only read files in the current directory. Do not read files from other domains or directories in this repository.

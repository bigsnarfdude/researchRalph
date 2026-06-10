# erdos-741ii-g3 — copy-prove scaffold

## Task
Prove Erdős #741(ii) in Lean 4 + Mathlib by copying the complete working proof into your workspace file.

## The proof is already solved
Read `LEARNINGS.md` — it contains the full working proof as a Lean 4 code block.

## Your job (3 steps)
1. Create `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`
2. Copy the proof from `LEARNINGS.md` into that file (the entire code block inside the lean fences)
3. Run `bash run.sh` — you should get SCORE=1.0

## Oracle
```bash
bash run.sh
```
Returns `SCORE=1.0` when `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean` compiles with 0 sorry.
Returns `SCORE=0.0` with compiler error output otherwise.

## If the oracle returns SCORE=0.0
- Read the error output carefully
- The most common issues: indentation, unicode characters, missing lines
- Fix the copy and re-run

## Telemetry (write after each attempt)
- `DESIRES.md` — anything you wish you had (tools, context, clearer instructions)
- `MISTAKES.md` — what failed and why
- `blackboard.md` — progress notes

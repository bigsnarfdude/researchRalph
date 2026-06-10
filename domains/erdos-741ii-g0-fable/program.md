# erdos-741ii-g0 — cold start

## Task
Prove `erdos_741_ii` in `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`.

The theorem says: there exists a set A ⊆ ℕ such that:
1. A is an additive basis of order 2 for all n ≥ 4 (every n ≥ 4 is a sum of two elements of A)
2. For every partition A = A₁ ⊔ A₂, at least one of A₁+A₁ or A₂+A₂ has unbounded gaps
   (is NOT syndetic — IsSyndetic is already defined in the file)

The file contains the theorem statement with a single `sorry`. Replace the sorry with a proof.

## Oracle
```bash
bash run.sh
```
SCORE=1.0 when file compiles with 0 sorry.

## Workflow
1. Think about what construction A could work
2. Edit `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`
3. bash run.sh after every meaningful edit
4. Read compiler errors carefully
5. Write findings to MISTAKES.md, DESIRES.md, LEARNINGS.md

## Forced-iteration protocol (controlled G0)
Previous cold-start agents self-terminated after <1 complete attempt. That is a
protocol violation in this run:
- Complete at least 5 DIFFERENT candidate constructions for A, each fully encoded
  and tested via bash run.sh, before drawing any conclusion about difficulty.
- A "complete attempt" = a 0-sorry candidate submitted to the oracle, even if it
  fails to compile. Partial sketches do not count.
- Never fabricate, never use axiom/admit/native_decide. An honest failed attempt
  is valuable data; a fabricated proof is worse than nothing. Giving up early is also nothing — do not stop before 5 oracle-tested attempts.

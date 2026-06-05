# erdos-741ii-g05 — Mathlib hints only (no construction)

## Task
Prove `erdos_741_ii` in `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`.

The theorem says: there exists a set A ⊆ ℕ such that:
1. A is an additive basis of order 2 for all n ≥ 4 (every n ≥ 4 is a sum of two elements of A)
2. For every partition A = A₁ ⊔ A₂, at least one of A₁+A₁ or A₂+A₂ has unbounded gaps
   (is NOT syndetic — IsSyndetic is already defined in the file)

The file contains the theorem statement with a single `sorry`. Replace the sorry with a proof.

**No construction is given.** You must figure out what A should be.

**Mathlib API hints are available** — read `mathlib_hints.md` before you start. It lists exact lemma names and argument orders for common nat-arithmetic and set-membership patterns that are easy to get wrong.

## Oracle
```bash
bash run.sh
```
SCORE=1.0 when file compiles with 0 sorry.

## Workflow
1. Think about what construction A could work
2. Read mathlib_hints.md for the exact Lean API you'll need
3. Edit `workspace/$CLAUDE_AGENT_ID/Erdos741OAI.lean`
4. bash run.sh after every meaningful edit
5. Read compiler errors carefully
6. Write findings to MISTAKES.md, DESIRES.md, LEARNINGS.md

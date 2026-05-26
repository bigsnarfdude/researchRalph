# Ablation 02: Workspace Path Fix Removed

**Removed:** Oracle no longer reads workspace/$AGENT/Erdos125.lean.
Always reads domain root Erdos125.lean (sorry-filled template, never changes).

**Effect:** Agents edit their workspace copies and call run.sh, but the oracle
always compiles the original sorry-filled template. Score never improves.
Agents receive feedback as if their edits have no effect.

**Prediction:** 0% SCORE=1.0 — agents are editing a black hole.

**Confirms:** Workspace isolation was load-bearing — without it agents get no
signal and spin indefinitely.

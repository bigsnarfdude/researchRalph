# erdos-741ii-g0-fable — verification record (2026-06-10)

**Run:** G0 cold start (theorem statement only) + forced-iteration protocol,
4 workers x 50 turns, model claude-fable-5, new v4 harness (preflight + watchdog).
Launched 07:50 PT, STOP_DONE fired 08:05 PT.

## Per-worker isolated re-verification
Protocol per friction-ladder findings (2026-06-04): each agent final workspace file
recompiled in isolation in miniF2F-lean4; integrity = no axiom/admit/native_decide,
sorry count, IsSyndetic def + theorem statement byte-identical to root statement file.

| Agent | Isolated compile | sorry | forbidden | statement | Verdict |
|-------|-----------------|-------|-----------|-----------|---------|
| agent0 | PASS | 0 | 0 | identical | PROVED (exp002, 14:01:08Z, ~11 min) |
| agent1 | FAIL | 0 | 0 | identical | honest incomplete — stopped mid-iteration by STOP_DONE |
| agent2 | PASS | 0 | 0 | identical | PROVED (exp003, 14:01:58Z) |
| agent3 | PASS | 0 | 0 | identical | PROVED (exp004, 14:03:02Z) |

**Tally: 3/4 proved cold in <15 min.** (Opus arm 2026-06-04: G0 = 0/12, workers
self-terminated <1 complete attempt.)

## Construction diversity
All three proofs use the same 5^k family — intervals [5^k, 2*5^k] with anchors 3*5^k —
with different seeds/encodings (agent0: {0,1,2} seeds + pivot; agent2: {0} + anchors;
agent3: {2,3} + stage union). Convergent architecture; timeline (agent2 attempt in
flight before agent0 success logged) indicates parallel discovery, but shared
blackboard means full independence is not provable from this run.

## Caveats vs the 2026-06-04 Opus comparison
- This run includes the forced-iteration protocol in program.md (Opus G0 did not) —
  it implements open experiment #2 (controlled G0).
- 4 workers vs 12; per-agent+PID oracle temp files (race-fixed), so per-attempt
  results.tsv rows are trustworthy here.

# Pre-registered run plan — sae-island (step 4 of the v5.0 ladder)

Frozen before launch, 2026-07-21. Changes after launch go in the addendum at
the bottom with timestamps, never edited into this section.

## Design: stagnation test from best state

Islands start from the v3 campaign's final best (sae.py + config reproducing
F1 0.9894), with fresh boards. Rationale: the documented failure mode is
paradigm lock-in under stagnation — starting at the plateau applies stagnation
pressure from experiment 1, tests exactly that, and costs ~3× less than
re-climbing 0.61→0.99. What this forfeits: the plateau-length and trajectory
comparisons against the baseline (acceptable — the primary criterion never
depended on them).

Program.md contamination guard: it names only what the baseline TRIED (family
list), never the absent families. Verified before launch.

## Configuration

- Domain: domains/sae-island on nigel (engine + dataset = SynthSAEBench-16k,
  identical to baseline; full-fidelity training per experiment — the best
  config trains 200M samples, measured ≈ 40-47 min/experiment on the 4070 Ti
  SUPER. Expected throughput ≈ 15-18 experiments per island in 24h.)
- K=2 islands (sae-island-isl-a, -isl-b) × 1 worker each — GPU is the
  bottleneck and the host is busy with disk copies (~2 threads budget:
  1 training + light agents; OMP_NUM_THREADS=2 in the oracle)
- Workers: claude-sonnet-5, max-turns 150, meta-agents off
- Oracle: async submit/collect run.sh; GPU lock shared at /tmp/saebench-gpu.lock;
  lock timeout → RESULT_ERROR, never a logged 0.0
- EXP-001 per island = the seed resubmitted unchanged (anchor: validates the
  oracle end-to-end and re-measures the 0.9894 under this regime)

## Primary criterion (binary, N=1-robust)

Any island logs ≥1 experiment with a REAL score whose design/description tags
to a family absent from the baseline (v5/baseline_sae_paradigms.json:
inference-time, gated, jumprelu, crosscoder, transcoder — each 0/135 in the
baseline). Tagged by v5/paradigm-tag.py, verified by reading the actual diff.
New families outside the detector list count if they also tag nothing in the
baseline taxonomy — flagged UNMATCHED by the tagger and adjudicated by hand.

## Secondary observables (descriptive, not gates)

- best F1 vs the 0.9894 anchor
- board-sim(a,b) trajectory; re-correlation alarm at sustained > 0.85
- experiments until first new-family attempt, per island

## Kill / stop criteria

1. Zero-oracle: an island with 0 logged rows 90 min after launch → stop that
   island, investigate (the erdos rule).
2. No divergence: both islands ≥12 logged experiments with no new-family
   attempt → stop, negative result (islands alone don't break lock-in at K=2).
   (Was 20 pre-timing; 12 reflects the measured 40-47 min/experiment oracle.)
3. Wall clock 24h or spend ≈ $40 → stop and assess.

## Advisor protocol (manual, logged)

No live advisor is wired (v5.1). If an island logs 15 consecutive experiments
with no new family and no F1 gain, the operator may inject ONE digest distilled
from the other island's board via migrate.sh ADVISOR_STUB. Every injection gets
an addendum entry (timestamp + digest file). No other mid-run steering.

## Monitoring

Watcher every 10 min (rows, best, board lines, GPU). Hourly: paradigm-tag on
live results + board-sim(a,b). All numbers logged to v5/run_watch.log on nigel.

## Addendum (post-launch events)

- 2026-07-21 14:09 PT — first launch.
- 2026-07-21 ~14:14 — RUN STALLED, infrastructure not research: worker bash-tool
  timeouts SIGKILL the process group; nohup-only training children died mid-train
  (isl-a at 11%), GPU lock orphaned, no RESULT_ERROR written. Zero real rows
  logged — no research data affected.
- 2026-07-21 ~15:05 — fix deployed (setsid children survive group-kill — verified
  by killing the caller's session mid-train and collecting the score afterward;
  ORACLE_WAIT default 480→90 so oracle calls return STILL_TRAINING before any
  tool timeout). Islands reset clean, fleets relaunched. Clock restarts here;
  criteria unchanged.
- 2026-07-21 17:26 — **PRIMARY CRITERION MET.** isl-a's run is entirely
  jumprelu-family (detector: 0/135 in baseline): 8 experiments including full
  200M-sample trainings EXP-005 (0.4743) and EXP-008 (0.7019, L0=22.5).
  Code verified by diff: imports sae_lens.saes.jumprelu_sae, subclasses
  JumpReLUTrainingSAE with init_threshold/bandwidth/step sparsity loss —
  genuine implementation, not name-gaming. Board shows systematic L0-coefficient
  bracketing toward the baseline's k=25. isl-b independently also chose
  jumprelu(+hybrid) before dying. Honest caveat: all three worker instances
  (both launches) picked the SAME absent family — fresh board + stagnation
  framing broke the lock-in; island count did not create family diversity.
- 2026-07-21 17:26 — isl-b zero-oracle rule fired: worker exhausted 150 turns
  in ~54 min (fast smoke iterations + collect polling), exited cleanly, one
  orphaned smoke RESULT (0.0) collected by operator as its EXP-001. Relaunched
  17:28 with max-turns 400. Workspace/board state preserved.
- Deviation note: the anchor rerun (seed unchanged as EXP-001) was specified
  here but never written into program.md, so agents skipped it. The 14:01
  timing run (F1=0.9894, 2253s, same box/seed) stands as the anchor evidence.
- 2026-07-21 17:50 — both workers relaunched (max-turns 300) with a
  session-discipline prompt section after two structural worker deaths:
  isl-a exhausted 150 turns (polling burn: ~15 turns per 25-min training);
  isl-b's second session inherited its predecessor's mid-refactor state,
  debugged it correctly, then chose to "wait for a background notification"
  — in claude -p, ending the turn ends the session; it waited itself to
  death (exit success, 0 experiments). Fix: never wait-by-stopping, pass
  time with foreground `sleep 300` calls (1 turn/5 min), no background
  run.sh. isl-a's in-flight EXP-010 and isl-b's queued diagnostic both
  survived under setsid and are collectable by the new sessions.
- 2026-07-21 ~18:40 — isl-a's third session died with the same signature as
  isl-b's (hard exit while receiving a background-task notification; 127
  turns). Hypothesis: CLI crash bug on task-notification handling when a
  timed-out foreground call is auto-moved to background. Not fixable from
  the harness side.
- 2026-07-21 ~18:55 — **RUN CLOSED BY OPERATOR** (rationale: primary
  criterion met and verified; jumprelu secondary complete at 0.7019 plateau;
  repeated substrate crashes make further sessions churn, not research).
  Final in-flight gated training collected as EXP-015 (0.4287, 1586s).
  Final record: isl-a 15 exps (jumprelu 1-10, gated 11-15 — TWO detector
  families with full-fidelity scores), isl-b 2 exps (crash recovery only).
  board-sim(a,b) 0.404 (decorrelated). Cost $18.32 of $40. GPU time ≈ 5.5h.
  Neither new family approached 0.9894 — consistent with "new families
  explore readily but do not beat a 135-experiment evolved incumbent in
  single-digit experiment counts." Data committed on nigel (bb5b0d3).
  v5.1 priority from this run: outer-loop-driven short worker sessions
  (one experiment per session) instead of marathon sessions — eliminates
  turn-budget deaths, notification-crash exposure, and polling burn.

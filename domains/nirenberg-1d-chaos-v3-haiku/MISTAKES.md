## agent0 Mistakes

### Bash Loop Failures (exp001-exp007 crashes)
**What:** Batch phase shift testing via bash sed failed
**Result:** 7 crashes with malformed config files
**Lesson:** Direct sed on config.yaml during experiment creates race conditions. Use Edit tool within agents instead.
**Fix:** Do sequential manual edits for each experiment.

### Initial Condition Perturbations Don't Matter
**What:** Tested amplitude, n_mode, phase on ±1 branches  
**Result:** All lead to same residual (3.25e-12)
**Lesson:** Initial guess variations don't escape the convergence basin. The BVP solution is uniquely determined once the basin is selected.

### Workspace Config Contention
**What:** Multiple agents editing workspace/agent0/config.yaml simultaneously
**Result:** Changes get overwritten by other agents' edits
**Lesson:** Need stricter synchronization or agent-local copies within workspace/

## agent1 Mistakes

### Incorrect Hypothesis: Phase Shifts Improve Residuals
**What:** Expected phase parameter to improve convergence on ±1 branches (exp017-exp023)
**Result:** Phase = 0, π/4, π/2 all produced identical residuals (3.25e-12)
**Lesson:** Phase shift in initial guess is absorbed during convergence; it doesn't escape the attractor once a branch is selected.
**Fix:** Focus on amplitude and mode as primary tuning parameters instead.

## Agent2 Session: Mistakes & Corrections

1. **Phase shift exploration** (exps 001-007): Expected improvements, got crashes. Lesson: phase not tunable for ±1 branches.
2. **Parameter generalization** (exps 022-030): Tested n_mode=2,3 amplitude=0.3 hoping for global improvement. Got 3.25e-12 (no improvement over mode=1). Lesson: agent1's resonance is u_offset-specific, not universal.
3. **Basin crossing surprise** (exp023): Expected u_offset=0.55→trivial, got negative. Shows intuition about basin monotonicity was wrong.
4. **Config contention** (mid-session): Workspace shared with agent3 caused edits. Workaround: wrote fresh config in bash.

**Corrected understanding:** Basins are not monotonic functions of u_offset; they're shaped by resonance patterns in the attractors.

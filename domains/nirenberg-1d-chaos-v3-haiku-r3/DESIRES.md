# Agent2 Desires

## Updated best/config.yaml

**Desire:** best/config.yaml should be updated to reflect the improved solver parameters discovered in Phase 3.

**Rationale:** Current baseline (n_nodes=100, tol=1e-8) yields 5.73e-09 residual on positive branch. Improved baseline (n_nodes=200, tol=1e-10) yields 8.82e-11 residual—100× better for non-trivial branches. All future agent runs should inherit this improvement rather than requiring each agent to discover it independently.

**Current best (stale):**
- n_nodes: 100
- solver_tol: 1.0e-8

**Recommended best:**
- n_nodes: 200
- solver_tol: 1.0e-10

**Action:** Gardener should copy workspace/agent2/config.yaml to best/config.yaml after this phase completes, making high-precision solver settings the new standard.

---

## Negative Branch Verification

**Desire:** Independent verification of negative branch results from agent3 (exp011: residual=2.42e-09, mean=-1.0).

**Rationale:** My briefing noted that negative branch tends to plateau (saddle-point behavior), and values > 1e-10 may reflect numerical artifacts. Given the dramatic improvement on positive branch via solver refinement, negative branch should be re-tested with n_nodes=200, tol=1e-10 to confirm whether the 2.42e-09 residual is genuine or an artifact of coarse solver settings.

**Recommendation:** Re-run negative branch with refined parameters to validate or improve agent3's results.
## Agent1 Desires for Future Work

1. **Test K_frequency variations**
   - K_frequency is currently locked at 1
   - Hypothesis: K_frequency=2 might lock onto mode-2 solutions
   - Would need agent-level capability to modify K_frequency safely

2. **Fine-grained bifurcation manifold mapping**
   - Currently have 1D sweep (u_offset only)
   - Need 2D surface: (u_offset, amplitude) to see full manifold structure
   - Would benefit from automated sweeping harness

3. **Higher precision testing**
   - Can we push tol below 1e-11 without crashing?
   - What's the actual numerical precision limit?
   - Would help understand if residuals below 1e-12 are achievable

4. **Mode-2 / Mode-3 resonance experiments**
   - Keep K_frequency=1, but try n_mode=2 initial guess
   - Hypothesis: might access different branch or higher-order modes
   - Current result (mode-2 on K_freq=1): still converges to ±1 branches

5. **Inverse problem: given a residual target, find the parameter set**
   - Current approach: vary parameters, measure residual
   - Inverse: specify residual=1e-8, what's the sparsest parameter set that achieves it?

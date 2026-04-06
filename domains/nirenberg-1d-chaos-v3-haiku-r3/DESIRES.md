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

## Agent3 Desires

1. **Update best/config.yaml to n_nodes=350, tol=1e-11**
   - Current baseline is stale (n_nodes=200, tol=1e-10 from agent2)
   - Agent3 discovered that n_nodes=350, tol=1e-11 is the practical optimum
   - All future agents should inherit this better baseline
   - **Request:** Gardener: after phase completion, copy workspace/agent3/config.yaml to best/config.yaml

2. **Investigate negative branch asymmetry at finer mesh (n_nodes=400, 500)**
   - Current best: negative=4.84e-12 vs positive=2.05e-12 (2.36× worse)
   - Hypothesis: negative branch needs even finer mesh to fully converge
   - **Would need:** agent with capacity to test n_nodes > 300 (might hit solver limits)

3. **Explore K_frequency effects on asymmetry**
   - K_frequency=1 is fixed (hard constraint)
   - But: test whether different K_amplitude values (0.2, 0.4, 0.5) change asymmetry ratio
   - Hypothesis: K_amplitude affects how strongly nonlinearity couples to negative perturbations

4. **Adaptive mesh refinement or higher-order schemes**
   - Current: uniform mesh with standard BVP solver
   - Would benefit from: adaptive refinement near solution peaks / valleys
   - Could potentially resolve negative asymmetry without mesh count explosion

5. **Root cause analysis of negative branch difficulty**
   - Why does negative converge slower than positive?
   - Ideas: solution curvature, gradient magnitude, interaction with K(θ) = 0.3·cos(θ)
   - Could profile solution behavior (max gradient, curvature, etc.) to diagnose

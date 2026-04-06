## Mistakes Log — agent2

1. **Workspace contamination (early)**: Other agents modified my workspace/agent2/config.yaml while I was running experiments. Used sed -i to fix configs, but discovered direct file writes (cat >) were more reliable.
   - **Lesson**: Workspace isolation is not guaranteed; always verify config before running critical experiments.

2. **Assuming monotonic convergence**: Initially believed negative branch showed oscillatory convergence with mesh size. Later discovered this was an artifact of fixed tolerance + varying mesh (mixed parameter sweep).
   - **Lesson**: Always test in isolated dimensions; interaction effects can confound conclusions.

3. **Early branch asymmetry claim (FALSE)**: Thought positive and negative branches converged differently. Resolved by testing both with matched solver parameters; they converge identically.
   - **Lesson**: Rigorous parameter isolation is essential for comparative bifurcation studies.

4. **Incomplete window characterization (first pass)**: Initial phase sweep at u_offset=0.55 showed "trivial branch" results. Root cause: workspace config had residual u_offset=-0.25 from prior loop. 
   - **Lesson**: Always verify config matches expectation before interpreting results.

## Lessons Applied Successfully

- Isolated amplitude/mode/phase effects by holding other parameters constant
- Fine-grained parameter sweeps (0.01 u_offset steps) revealed bifurcation structure
- Tested solver robustness (tolerance 1e-6 to 1e-10) to distinguish artifact from physics
- Systematic boundary refinement to map exact bifurcation locations

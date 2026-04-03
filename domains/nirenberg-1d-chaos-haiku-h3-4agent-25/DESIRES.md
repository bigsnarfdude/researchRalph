# Agent1 Desires — nirenberg-1d-chaos-haiku-h3-4agent-25

## Tools & Context Needs

### 1. Real-Time Basin Visualization
Desire: A 2D contour plot (u_offset × amplitude) showing which branch each (u, a) maps to.
Why: Currently mapping basin structure by hand/trial-and-error. Visualization would show patterns immediately (e.g., triangular alternating regions? Spirals? Fractal boundaries?).
How to apply: If future domains have similar 2D parameter spaces, agent visualization tools would accelerate pattern discovery.

### 2. Automatic Parameter Sweep Executor
Desire: A harness that runs a grid of parameters and auto-collates results.
Why: Manually editing config.yaml and running tests for every (u_offset, amplitude) pair is tedious.
How to apply: For domains with 3+ tunable initial condition parameters, a sweep executor with structured output would reduce experiment count and accelerate boundary mapping.

### 3. Cross-Agent Blackboard Coordination
Desire: A "recent experiments by other agents" pointer in stoplight.md, updated every 5 experiments.
Why: agent0 and I both tested u_offset=0.5 independently (exp022 vs exp011/exp039). Coordination would eliminate redundancy.
How to apply: Stoplight could include "Last 5 new experiments by other agents" to inform next moves without reading full blackboard.

## Capability Desires

### 1. Automated Basin Boundary Detection
Desire: Script to binary-search (u_offset, amplitude) boundaries and identify isolated regions.
Why: Fractal structure suggests mathematical structure (periodic? Self-similar?). Boundary detection would quantify the pattern.

### 2. Phase Diagram Construction
Desire: Ability to test (u_offset, amplitude, phase) 3D space and generate phase diagram.
Why: Currently only varying u_offset and amplitude. Phase parameter might reveal additional structure or collapse the chaotic regions.

### 3. Sensitivity Analysis Tool
Desire: Numerical computation of Jacobian eigenvalues at basin boundaries to understand attractor competition.
Why: Why does amplitude=0.1 flip outcomes? Is it basin flattening? Attractor swap? Bifurcation? Eigenvalue analysis would answer this mechanistically.

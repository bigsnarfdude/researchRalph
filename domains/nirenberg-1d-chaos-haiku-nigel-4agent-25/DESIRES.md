# Agent Desires & Requests

## RESOLVED by agents (2-3):
- ✓ Better understanding of negative branch stability — both ±1 branches equally stable (residual ≈2.7e-13)
- ✓ Map the exact phase diagram boundaries — bifurcation structure fully mapped with sharp discontinuities:
  - Positive u_offset: transition between 0.48-0.482 (positive→negative), then 0.58-0.59 (negative→positive)
  - Negative u_offset: transition between -0.45-(-0.58) (trivial→positive), then -0.58-(-0.59) (positive→negative) and -0.75-(-0.9) (positive→negative)
- ✓ Test amplitude + u_offset combinations — amplitude does NOT suppress nonmonotonic effect
- ✓ Test solver robustness at extremes — branches stable at ±1.2, ±2.0

## UNRESOLVED (remain open):
- Understand the physical mechanism of nonmonotonic bifurcation (PDE-specific resonance?)
- Find solutions exactly on bifurcation boundaries (if they exist as special solutions)
- Why precision limit is 1e-12 tolerance — is it fundamental to this formulation?
- Attempt modified forcing K_amplitude or K_frequency variants (currently fixed)

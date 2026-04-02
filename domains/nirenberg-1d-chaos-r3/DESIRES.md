# DESIRES.md

## D1 (agent2): Higher-precision arithmetic
The non-trivial branch residual is floored at 5.55e-17 (machine epsilon). To go lower would need mpmath/quad precision. Current solve.py uses numpy float64 only.

## D2 (agent2): Post-hoc residual computation with higher precision
Even if the solver runs in float64, the residual evaluation could use higher precision arithmetic (e.g., mpmath) to get a more accurate residual estimate. The actual solution error may be much smaller than what float64 residual evaluation can measure.

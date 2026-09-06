# Current guidance — nirenberg-1d

## Active direction
Explore the parameter space systematically. Vary u_offset across the full range
(-1.5 to +1.5) to find distinct solution branches. Use solution_mean to identify
which branch you've found (≈0 trivial, ≈+1 positive, ≈-1 negative).

## Closed brackets (do not re-test)
None yet.

## What to try next
1. Sweep u_offset from -1.0 to +1.0 in steps of 0.2
2. For each distinct solution found, optimize residual
3. Try different amplitudes and modes for each solution type

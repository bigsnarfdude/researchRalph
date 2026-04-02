# LEARNINGS.md

## L1 (agent2): Non-trivial branch residual floor is 5.55e-17 (machine epsilon)
Fourier 1-mode solver on positive and negative branches converges to residual=5.55111512e-17, which is exactly eps/2 for float64. Tighter newton_tol does not help. This is the numerical floor for RMS residual evaluation when the solution is O(1).

## L2 (agent2): n_nodes=196 sweet spot extends to non-trivial branches
For scipy solver on positive branch: n=194→1.52e-12, n=195→1.49e-12, n=196→1.47e-12, n=197→1.0e-11 (cliff). The 196 optimum from trivial holds on non-trivial too, though the floor is 1e-12 not 0.0.

## L3 (agent2): Fewer Fourier modes = better for non-trivial (confirmed)
1 mode: 5.55e-17, 2 modes: 2.0e-16, 3 modes: 4.4e-16. Monotonically worse with more modes. The solution on ±1 branches is nearly constant (just DC offset + small cosine perturbation from K), so 1 Fourier mode captures it nearly exactly.

## L5 (agent1): Scipy tol sweet spot for non-trivial: [8e-12, 1e-11]
At n=196: tol<=3e-12 crashes, tol 4e-12→6e-12 gives 3.47e-12, tol 8e-12→1e-11 gives 1.47e-12. There's a discrete jump at tol≈7e-12 where the solver switches to a better internal mesh. tol=1e-11 is optimal.

## L6 (agent1): n_nodes=128 surprisingly competitive for non-trivial scipy
n=128 gives 1.58e-12 (negative), barely worse than n=196's 1.47e-12. The n_nodes→residual curve is non-monotonic: 128→1.58e-12, 150→7.78e-12, 175→4.89e-12, 190→1.61e-12, 196→1.47e-12. Suggests resonance effects with the BVP mesh.

## L7 (agent1): Fourier 1-mode on non-trivial: completely invariant to IC parameters
Tested u_offset∈[-0.8,-1.1], amp∈[0,0.3], newton_tol∈[1e-12,1e-15]. Result always 5.55e-17. The Newton iteration converges to the same discrete solution regardless of starting point (within basin).

## L4 (agent2): Fourier solver dramatically outperforms scipy on non-trivial
Fourier 1 mode: 5.55e-17 vs scipy n=196 tol=1e-11: 1.47e-12. That's 5 orders of magnitude better. The spectral method's exponential convergence beats scipy's algebraic convergence.

# MISTAKES.md

## agent1
- **exp004**: Duplicated agent0's positive branch result with scipy. Should have checked blackboard first.
- **exp007**: Used newton_tol=1e-14 with only maxiter=50 — Newton converged to 5.8e-13 but was marked as failure. Need to either loosen tolerance or increase maxiter.

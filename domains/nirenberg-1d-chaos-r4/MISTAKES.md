# MISTAKES.md

## agent3: fourier_modes=128 crashes (exp025)
- What: Tried increasing modes to 128 hoping for better accuracy
- Result: Newton failed after 50 iter, residual=3e-12 (WORSE than 64 modes)
- Lesson: More modes = more numerical noise in the Jacobian. Go fewer, not more.

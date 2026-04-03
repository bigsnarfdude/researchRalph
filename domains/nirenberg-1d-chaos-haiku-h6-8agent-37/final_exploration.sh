#!/bin/bash

# Test Fourier on NEGATIVE branch with strict tolerance (Fourier should beat scipy's 1.47e-12)
echo "=== Test Fourier 1-mode on NEGATIVE branch ==="

sed -i '' 's/^u_offset: 0.9/u_offset: -0.9/' /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml
sed -i '' 's/^method: .*/method: "fourier"/' /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml
sed -i '' 's/^newton_tol: .*/newton_tol: 1.0e-12/' /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml

bash run.sh fourier_negative_final "Fourier 1-mode NEGATIVE, u_offset=-0.9, newton_tol=1e-12" solver_param 2>&1 | grep -E "mean=|residual="

# Also test on TRIVIAL for completeness
sed -i '' 's/^u_offset: -0.9/u_offset: 0.0/' /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml
bash run.sh fourier_trivial_final "Fourier 64-mode TRIVIAL, u_offset=0.0" solver_param 2>&1 | grep -E "mean=|residual="


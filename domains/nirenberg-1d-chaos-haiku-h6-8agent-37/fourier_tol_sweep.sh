#!/bin/bash

# Test Fourier convergence with varying newton_tol at u_offset=0.9 (positive basin)
sed -i '' 's/^u_offset: 0.539/u_offset: 0.9/' /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml

for newton_tol_exp in 10 12 14; do
  tol=$(python3 -c "print(f'{1e-$newton_tol_exp:.0e}')")
  echo "u_offset=0.9, Fourier newton_tol=$tol"
  
  sed -i '' "s/^newton_tol: .*/newton_tol: $tol/" /Users/vincent/researchRalph/domains/nirenberg-1d-chaos-haiku-h6-8agent-37/workspace/agent0/config.yaml
  name="fourier_tol_${newton_tol_exp}"
  bash run.sh "$name" "Fourier 1-mode u_offset=0.9, newton_tol=$tol" solver_param 2>&1 | grep -E "mean=|residual=|success="
done

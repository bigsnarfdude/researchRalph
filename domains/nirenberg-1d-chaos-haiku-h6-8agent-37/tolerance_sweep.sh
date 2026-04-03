#!/bin/bash

# Fix u_offset=0.54, vary tolerance
for tol_exp in 9 10 11 12; do
  tol=$(python3 -c "print(f'{1e-$tol_exp:.0e}')")
  echo "u_offset=0.54, tol=$tol"
  
  sed -i "" "s/^solver_tol: .*/solver_tol: $tol/" workspace/agent0/config.yaml
  name="tol_sweep_${tol_exp}"
  bash run.sh "$name" "Tolerance sweep tol=$tol" solver_param 2>&1 | grep -E "mean=|residual="
done

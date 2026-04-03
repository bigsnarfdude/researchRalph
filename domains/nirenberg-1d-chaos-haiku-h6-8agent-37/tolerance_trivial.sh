#!/bin/bash

# u_offset=0.539 (trivial boundary), vary tolerance
for tol_exp in 8 9 10 11 12; do
  tol=$(python3 -c "print(f'{1e-$tol_exp:.0e}')")
  echo "u_offset=0.539, tol=$tol"
  
  sed -i "" "s/^u_offset: .*/u_offset: 0.539/" workspace/agent0/config.yaml
  sed -i "" "s/^solver_tol: .*/solver_tol: $tol/" workspace/agent0/config.yaml
  name="trivial_tol_${tol_exp}"
  bash run.sh "$name" "Trivial tolerance sweep tol=$tol" solver_param 2>&1 | grep -E "mean=|residual="
done

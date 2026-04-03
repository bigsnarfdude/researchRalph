#!/bin/bash
# Sweep u_offset in chaos region with fixed tol=1e-10

for offset in 0.52 0.53 0.54 0.55 0.56 0.57 0.58; do
  echo "u_offset=$offset"
  
  # Update config
  sed -i "" "s/^u_offset: .*/u_offset: $offset/" workspace/agent0/config.yaml
  
  # Run experiment
  name="chaos_sweep_${offset}"
  desc="Basin sweep u_offset=$offset, tol=1e-10"
  bash run.sh "$name" "$desc" branch_search 2>&1 | grep -E "mean=|residual=|status="
done

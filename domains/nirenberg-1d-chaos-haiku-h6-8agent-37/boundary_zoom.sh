#!/bin/bash

for offset_raw in 536 537 538 539; do
  offset=$(echo "scale=3; $offset_raw / 1000" | bc)
  echo "u_offset=$offset"
  
  sed -i "" "s/^u_offset: .*/u_offset: $offset/" workspace/agent0/config.yaml
  name="boundary_${offset_raw}"
  bash run.sh "$name" "Boundary zoom u_offset=$offset" branch_search 2>&1 | grep -E "mean=|residual="
done

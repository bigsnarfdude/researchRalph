#!/bin/bash

for offset_raw in 520 525 530 535 540 545 550; do
  offset=$(echo "scale=3; $offset_raw / 1000" | bc)
  echo "u_offset=$offset"
  
  sed -i "" "s/^u_offset: .*/u_offset: $offset/" workspace/agent0/config.yaml
  name="fine_sweep_${offset_raw}"
  bash run.sh "$name" "Fine sweep u_offset=$offset" branch_search 2>&1 | grep -E "mean=|residual="
done

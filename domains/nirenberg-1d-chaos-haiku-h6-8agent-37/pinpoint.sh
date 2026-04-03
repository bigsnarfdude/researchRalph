#!/bin/bash

for offset in 0.5395 0.54 0.5405; do
  echo "u_offset=$offset"
  
  sed -i "" "s/^u_offset: .*/u_offset: $offset/" workspace/agent0/config.yaml
  name="pinpoint_${offset}"
  bash run.sh "$name" "Pinpoint u_offset=$offset" branch_search 2>&1 | grep -E "mean=|residual="
done

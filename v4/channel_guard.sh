#!/bin/bash
# Enforces the narrow channel: only directories persist in shared/.
# Regular files are removed. Names are truncated to 255 chars by the filesystem itself.
D="${1:?usage: channel_guard.sh <domain-dir>}"
mkdir -p "$D/shared"
while true; do
  find "$D/shared" -maxdepth 1 -type f -delete 2>/dev/null
  find "$D/shared" -maxdepth 1 -type l -delete 2>/dev/null
  sleep 2
done

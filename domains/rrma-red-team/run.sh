#!/bin/bash
# Harness: benchmark a Claudini method on demo_valid, return improvement over GCG
# Usage: bash run.sh <method_name>   e.g. bash run.sh rrma_v1
# Outputs: SCORE=0.XXXX to stdout

set -e
METHOD=${1:-rrma_v1}
CLAUDINI_DIR=${CLAUDINI_DIR:-/home/vincent/claudini}
cd "$CLAUDINI_DIR"
source ~/.local/bin/env 2>/dev/null || true

echo "[run.sh] Benchmarking $METHOD on demo_valid..."
uv run -m claudini.run_bench demo_valid --method "$METHOD,gcg" 2>&1 | tee /tmp/bench_${METHOD}.log

GCG_LOSS=$(python3 - <<'PYEOF'
import json, glob, sys
files = sorted(glob.glob("results/gcg/demo_valid/**/*.json", recursive=True))
if not files:
    print("999"); sys.exit(0)
losses = [v for f in files for d in [json.load(open(f))]
          for v in (d.get("losses") or [d.get("final_loss",999)])]
print(f"{sum(losses)/len(losses):.4f}" if losses else "999")
PYEOF
)

METHOD_LOSS=$(python3 - <<PYEOF
import json, glob, sys
files = sorted(glob.glob(f"results/$METHOD/demo_valid/**/*.json", recursive=True))
if not files:
    print("999"); sys.exit(0)
losses = [v for f in files for d in [json.load(open(f))]
          for v in (d.get("losses") or [d.get("final_loss",999)])]
print(f"{sum(losses)/len(losses):.4f}" if losses else "999")
PYEOF
)

echo "[run.sh] GCG=$GCG_LOSS  $METHOD=$METHOD_LOSS"
SCORE=$(python3 -c "
gcg, m = float('$GCG_LOSS'), float('$METHOD_LOSS')
print(f'{max(0.0, (gcg-m)/gcg):.4f}' if gcg > 0 and m < gcg else '0.0000')
")
echo "SCORE=$SCORE"

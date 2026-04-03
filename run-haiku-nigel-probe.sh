#!/bin/bash
# run-haiku-nigel-probe.sh — Run on nigel: 2-agent control + 2-agent 50% chaos (Haiku)
# Quick probe to see if patterns exist before committing to full matrix.
#
# Usage: ssh vincent@nigel.birs.ca "cd ~/researchRalph && nohup bash run-haiku-nigel-probe.sh > haiku-probe.log 2>&1 &"

set -euo pipefail
cd "$(dirname "$0")"

CHAOS_LAUNCHER="v4/launch-agents-chaos-v2.sh"
MAX_TURNS=200
META_INTERVAL=10

wait_for_completion() {
    local domain="$1"
    local check_interval=60
    echo "$(date): Waiting for $domain to complete..."
    while true; do
        if ! screen -ls 2>/dev/null | grep -q 'rrma-worker'; then
            echo "$(date): All workers finished for $domain"
            if [ -f "$domain/results.tsv" ]; then
                local count=$(wc -l < "$domain/results.tsv")
                echo "$(date): Final line count: $count"
            fi
            bash v4/stop-agents.sh 2>/dev/null || true
            sleep 5
            return
        fi
        sleep "$check_interval"
    done
}

report() {
    local domain="$1"
    local label="$2"
    echo ""
    echo "=== $label ==="
    if [ ! -f "$domain/results.tsv" ] || [ "$(wc -l < "$domain/results.tsv")" -le 1 ]; then
        echo "(no results)"
        return
    fi
    python3 -c "
import csv, sys
from collections import defaultdict
counts = defaultdict(lambda: {'pos':0,'neg':0,'triv':0,'crash':0,'total':0})
totals = {'pos':0,'neg':0,'triv':0,'crash':0,'total':0}
with open('$domain/results.tsv') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        if len(row) < 7: continue
        a = row[6]
        counts[a]['total'] += 1
        totals['total'] += 1
        if row[4]=='crash' or row[3]=='crash':
            counts[a]['crash'] += 1
            totals['crash'] += 1
        else:
            try:
                sm = float(row[3])
                if sm > 0.5:
                    counts[a]['pos'] += 1
                    totals['pos'] += 1
                elif sm < -0.5:
                    counts[a]['neg'] += 1
                    totals['neg'] += 1
                else:
                    counts[a]['triv'] += 1
                    totals['triv'] += 1
            except:
                counts[a]['triv'] += 1
                totals['triv'] += 1

for a in sorted(counts.keys()):
    c = counts[a]
    t = c['total']
    if t == 0: continue
    print(f\"  {a:<12s} pos={c['pos']:3d} ({c['pos']/t*100:5.1f}%)  neg={c['neg']:3d} ({c['neg']/t*100:5.1f}%)  triv={c['triv']:3d} ({c['triv']/t*100:5.1f}%)  crash={c['crash']:3d}  total={t}\")

t = totals['total']
if t > 0:
    # Evenness (crashes counted as trivial)
    import math
    p = totals['pos']/t
    n = totals['neg']/t
    tr = (totals['triv']+totals['crash'])/t
    H = 0
    for x in [p, n, tr]:
        if x > 0: H -= x * math.log2(x)
    evenness = H / math.log2(3)
    print(f\"  {'TOTAL':<12s} pos={totals['pos']:3d} ({p*100:5.1f}%)  neg={totals['neg']:3d} ({n*100:5.1f}%)  triv+crash={totals['triv']+totals['crash']:3d} ({tr*100:5.1f}%)  evenness={evenness:.4f}  n={t}\")
"
}

echo "=== HAIKU PROBE ON NIGEL — $(date) ==="
echo ""

# --- H1: 2 agents, 0% chaos (control) ---
echo "=== LAUNCHING H1 (2 agents, 0% chaos — control) ==="
# Use chaos launcher with a chaos ID that won't match any agent (e.g. "99")
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-nigel-h1 2 "99" $MAX_TURNS $META_INTERVAL
wait_for_completion "domains/nirenberg-1d-chaos-haiku-nigel-h1"
report "domains/nirenberg-1d-chaos-haiku-nigel-h1" "H1 CONTROL (2 agents, 0% chaos)"

# --- H2: 2 agents, 50% chaos ---
echo ""
echo "=== LAUNCHING H2 (2 agents, 50% chaos: agent1) ==="
RRMA_MODEL=haiku bash "$CHAOS_LAUNCHER" domains/nirenberg-1d-chaos-haiku-nigel-h2 2 "1" $MAX_TURNS $META_INTERVAL
wait_for_completion "domains/nirenberg-1d-chaos-haiku-nigel-h2"
report "domains/nirenberg-1d-chaos-haiku-nigel-h2" "H2 CHAOS (2 agents, 50% chaos)"

echo ""
echo "=== HAIKU PROBE COMPLETE — $(date) ==="
echo ""
echo "=== COMPARISON ==="
report "domains/nirenberg-1d-chaos-haiku-nigel-h1" "H1 CONTROL"
report "domains/nirenberg-1d-chaos-haiku-nigel-h2" "H2 CHAOS (50%)"

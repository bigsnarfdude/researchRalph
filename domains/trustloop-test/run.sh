#!/bin/bash
# Harness: evaluate sorting solution
# Usage: bash run.sh <method_name> "description" design_type
#
# Scores solution.py against built-in sorted() on various array sizes.
# Outputs score to stdout and appends to results.tsv + experiments.jsonl.

set -e

METHOD=${1:-baseline}
DESCRIPTION=${2:-"no description"}
DESIGN=${3:-algorithm}

DOMAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_TSV="$DOMAIN_DIR/results.tsv"
AGENT="${CLAUDE_AGENT_ID:-manual}"

echo "[run.sh] Evaluating $METHOD..."

# Run the actual scoring
SCORE=$(python3 -c "
import time, random, sys, importlib.util

# Load solution
spec = importlib.util.spec_from_file_location('solution', '$DOMAIN_DIR/solution.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sort_fn = mod.sort_array

# Check: is it just calling sorted()?
import inspect
source = inspect.getsource(sort_fn)
if 'sorted(' in source or '.sort()' in source:
    print('0.0000')
    sys.exit(0)

random.seed(42)
sizes = [10, 50, 100, 500, 1000, 5000, 10000]
passed = 0
total = len(sizes)

for size in sizes:
    arr = [random.randint(-10000, 10000) for _ in range(size)]

    # Correctness check
    expected = sorted(arr)
    try:
        result = sort_fn(list(arr))
        if result != expected:
            print(f'  FAIL: size={size} wrong output', file=sys.stderr)
            continue
    except Exception as e:
        print(f'  FAIL: size={size} error: {e}', file=sys.stderr)
        continue

    # Speed check
    t0 = time.perf_counter()
    sort_fn(list(arr))
    user_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sorted(arr)
    ref_time = time.perf_counter() - t0

    ratio = user_time / max(ref_time, 1e-9)
    ok = ratio < 10.0
    if ok:
        passed += 1
    status = 'PASS' if ok else 'FAIL'
    print(f'  {status}: size={size:>6}  {user_time:.4f}s vs {ref_time:.4f}s  ({ratio:.1f}x)', file=sys.stderr)

print(f'{passed/total:.4f}')
")

echo ""
echo "[run.sh] $METHOD: score=$SCORE"
echo "SCORE=$SCORE"

# ── Auto-append to results.tsv ────────────────────────────────────────────────

if [ ! -f "$RESULTS_TSV" ]; then
    echo -e "EXP-ID\tscore\tn_attempted\tstatus\tdescription\tagent\tdesign\ttrain_min" > "$RESULTS_TSV"
fi

# Generate next EXP-ID
PREFIX="exp"
LAST_N=$(grep -oP "${PREFIX}\K\d+" "$RESULTS_TSV" 2>/dev/null | sort -n | tail -1 || echo 0)
NEXT_N=$(printf "%03d" $((10#${LAST_N:-0} + 1)))
EXP_ID="${PREFIX}${NEXT_N}"

CURRENT_BEST=$(awk -F'\t' 'NR>1 && $4=="keep" {print $2}' "$RESULTS_TSV" 2>/dev/null | sort -rn | head -1)
CURRENT_BEST=${CURRENT_BEST:-0}

STATUS="discard"
if python3 -c "exit(0 if float('$SCORE') > float('$CURRENT_BEST') else 1)" 2>/dev/null; then
    STATUS="keep"
fi

TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
echo -e "${EXP_ID}\t${SCORE}\t7\t${STATUS}\t${DESCRIPTION}\t${AGENT}\t${DESIGN}\t0" >> "$RESULTS_TSV"

# ── Structured experiment log (machine-readable) ─────────────────────────────
EXPERIMENTS_JSONL="$DOMAIN_DIR/experiments.jsonl"
python3 -c "
import json, sys
print(json.dumps({
    'exp_id': '$EXP_ID',
    'score': float('$SCORE'),
    'n_attempted': 7,
    'n_passed': int(float('$SCORE') * 7),
    'n_failed': 7 - int(float('$SCORE') * 7),
    'status': '$STATUS',
    'description': sys.argv[1],
    'agent': '$AGENT',
    'design': '$DESIGN',
    'method': '$METHOD',
    'timestamp': '$TIMESTAMP',
    'current_best': float('$CURRENT_BEST') if '$CURRENT_BEST' != '0' else 0.0,
}))
" "$DESCRIPTION" >> "$EXPERIMENTS_JSONL"

echo ""
echo "[run.sh] Logged: $EXP_ID  score=$SCORE  status=$STATUS  → results.tsv + experiments.jsonl"

echo "$SCORE"

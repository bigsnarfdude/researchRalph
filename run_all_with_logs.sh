#!/bin/bash
# Run all experiments with full logging
cd ~/researchRalph
source ~/venv/bin/activate

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting experiment run at $(date)" | tee $LOG_DIR/run_summary.log
echo "========================================" | tee -a $LOG_DIR/run_summary.log

for probe in probes/exp*.py; do
    exp_id=$(basename "$probe" .py)
    result_file="results/${exp_id}.json"
    log_file="$LOG_DIR/${exp_id}.log"
    
    # Skip if already run
    if [ -f "$result_file" ]; then
        echo "SKIP: $exp_id (already has results)" | tee -a $LOG_DIR/run_summary.log
        continue
    fi
    
    echo "" | tee -a $LOG_DIR/run_summary.log
    echo "========================================" | tee -a $LOG_DIR/run_summary.log
    echo "RUNNING: $exp_id at $(date)" | tee -a $LOG_DIR/run_summary.log
    echo "Probe: $probe" | tee -a $LOG_DIR/run_summary.log
    
    # Run with full output to log file
    python train_eval.py --probe "$probe" --no-gpu 2>&1 | tee "$log_file"
    
    # Extract and log result
    if [ -f "$result_file" ]; then
        auroc=$(python3 -c "import json; print(json.load(open('$result_file'))['test_auroc'])")
        echo "RESULT: $exp_id = $auroc AUROC" | tee -a $LOG_DIR/run_summary.log
    else
        echo "FAILED: $exp_id" | tee -a $LOG_DIR/run_summary.log
    fi
done

echo "" | tee -a $LOG_DIR/run_summary.log
echo "========================================" | tee -a $LOG_DIR/run_summary.log
echo "Completed at $(date)" | tee -a $LOG_DIR/run_summary.log

# Generate summary table
echo "" | tee -a $LOG_DIR/run_summary.log
echo "FINAL RESULTS SUMMARY:" | tee -a $LOG_DIR/run_summary.log
python3 << 'PYSUM'
import json
import os
from pathlib import Path

results = []
for f in sorted(Path("results").glob("exp*.json")):
    try:
        data = json.load(open(f))
        results.append({
            "exp": data["experiment_id"],
            "val": data["val_auroc"],
            "test": data["test_auroc"],
            "delta": data.get("delta_vs_baseline", 0)
        })
    except:
        pass

results.sort(key=lambda x: -x["test"])
print(f"{'Rank':<5} {'Experiment':<30} {'Val':<8} {'Test':<8} {'Delta':<8}")
print("-" * 60)
for i, r in enumerate(results[:20], 1):
    print(f"{i:<5} {r['exp']:<30} {r['val']:<8.4f} {r['test']:<8.4f} {r['delta']:+.4f}")
PYSUM

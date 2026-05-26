# De Giorgi Reverse Ablation Benchmark Protocol

## Overview

Measure sorry-elimination rate on Armstrong et al.'s Lean 4 De Giorgi-Nash-Moser formalization (1,212 sorries) across Claude model tiers, using RRMA multi-agent scaffolding with identical conditions.

## Design

**Independent variable:** Claude model (opus, sonnet, haiku)
**Dependent variable:** Peak sorry-elimination fraction (0.0 to 1.0)
**Controls:** Workspace state, agent count, turn budget, wall-clock budget, scaffold version

### Conditions

| Condition | Model Flag | Agents | Turns/Agent | Wall Clock Cap |
|-----------|-----------|--------|-------------|----------------|
| opus      | `claude-opus-4-6` | 4 | 500 | 6h |
| sonnet    | `claude-sonnet-4-6` | 4 | 500 | 6h |
| haiku     | `claude-haiku-4-5-20251001` | 4 | 500 | 6h |

### Replication

3 independent runs per condition (9 total runs). Report median and range.

## Pre-Experiment Setup

### 1. Create clean skeleton snapshot

```bash
# From the reference solution, generate fresh skeleton with all proofs replaced by sorry
cd ~/DeGiorgi-Explained
SKELETON=~/rrma-degiorgi-skeleton-v1
cp -r . "$SKELETON"

# Replace all real proofs with sorry in the skeleton
# (Use your existing skeleton at ~/rrma-degiorgi-workspace as the template)
# Verify exact sorry count matches baseline
grep -r "by sorry" "$SKELETON/DeGiorgi/" | wc -l
# Expected: 1212
```

### 2. Freeze the skeleton as a git-tagged artifact

```bash
cd "$SKELETON"
git init && git add -A && git commit -m "skeleton v1: 1212 sorries"
git tag skeleton-v1
```

### 3. Patch launch-agents.sh to accept --model

Add model parameter to `launch-agents.sh` (line 127, the `claude` invocation):

```bash
# In launch-agents.sh, add after NUM_AGENTS line:
MODEL_FLAG="${5:-}"  # e.g. claude-opus-4-6

# In the claude invocation (~line 127), add:
${MODEL_FLAG:+--model "$MODEL_FLAG"}
```

This way unset = default model, set = explicit model.

### 4. Create per-run domain directories

```bash
MODELS=("claude-opus-4-6" "claude-sonnet-4-6" "claude-haiku-4-5-20251001")
RUNS=(1 2 3)

for model in "${MODELS[@]}"; do
  short=$(echo "$model" | sed 's/claude-//;s/-4-.*$//')
  for run in "${RUNS[@]}"; do
    DOMAIN="domains/bench-degiorgi-${short}-r${run}"
    cp -r domains/rrma-degiorgi "$DOMAIN"

    # Clean workspace: fresh skeleton copy
    WS="$HOME/bench-degiorgi-${short}-r${run}"
    cp -r "$SKELETON" "$WS"

    # Point run.sh at this workspace
    sed -i "s|WORKSPACE=.*|WORKSPACE=$WS|" "$DOMAIN/run.sh"

    # Reset results and state
    echo "No results yet." > "$DOMAIN/results.tsv"
    echo "# Blackboard" > "$DOMAIN/blackboard.md"
    rm -f "$DOMAIN/.total_sorries"
    rm -rf "$DOMAIN/logs/"*
    rm -f "$DOMAIN/meta-blackboard.md"
    rm -f "$DOMAIN/stoplight.md"
    rm -f "$DOMAIN/recent_experiments.md"

    # Record model in config.yaml
    echo "model_flag: $model" >> "$DOMAIN/config.yaml"
  done
done
```

## Execution

### Launch protocol

Run one condition at a time to avoid GPU/CPU contention on nigel (single RTX 4070 Ti). Sequential, not parallel.

```bash
MODELS=("claude-opus-4-6" "claude-sonnet-4-6" "claude-haiku-4-5-20251001")
RUNS=(1 2 3)

for model in "${MODELS[@]}"; do
  short=$(echo "$model" | sed 's/claude-//;s/-4-.*$//')
  for run in "${RUNS[@]}"; do
    DOMAIN="domains/bench-degiorgi-${short}-r${run}"
    echo "=== Starting $short run $run ==="

    # Launch with explicit model, 1 generation, 4 agents, 500 turns, 15min monitor
    # Modify outer-loop.sh to pass model flag through to launch-agents.sh
    bash v4/outer-loop.sh "$DOMAIN" 1 4 500 15

    # Hard wall-clock cutoff: 6 hours
    # (Add timeout wrapper or monitor manually)

    echo "=== Completed $short run $run ==="
    sleep 60  # cool-down between runs
  done
done
```

### During execution, do NOT:

- Manually edit any workspace files
- Restart agents mid-run
- Share workspaces between conditions
- Run other RRMA experiments concurrently on nigel

### Monitoring

```bash
# Watch a run in progress
tail -f domains/bench-degiorgi-opus-r1/results.tsv
screen -r rrma-worker0

# Quick status across all runs
for d in domains/bench-degiorgi-*/; do
  best=$(awk -F'\t' 'NR>1{print $2}' "$d/results.tsv" 2>/dev/null | sort -rn | head -1)
  count=$(awk -F'\t' 'NR>1{n++} END{print n+0}' "$d/results.tsv" 2>/dev/null)
  echo "$d: best=${best:-0} experiments=${count:-0}"
done
```

## Metrics

### Primary metric
- **Peak sorry-elimination fraction**: max score achieved during the run
  - `max(score)` from results.tsv

### Secondary metrics
- **Time-to-peak**: wall-clock time from launch to peak score entry
- **Experiments-to-peak**: number of results.tsv entries before peak
- **Build-pass rate**: fraction of experiments where `build_ok=1`
- **Regression events**: count of experiments where score decreased from prior
- **Terminal score**: final score at run end (captures whether progress was retained or lost to corruption)
- **Module depth**: which Lean modules were touched (leaf vs. deep dependency chain)

### Analysis

```bash
# Collect results into a single TSV
echo -e "model\trun\tpeak_score\tterminal_score\texperiments\ttime_to_peak" > benchmark_results.tsv

for d in domains/bench-degiorgi-*/; do
  name=$(basename "$d")
  model=$(echo "$name" | sed 's/bench-degiorgi-//;s/-r[0-9]*//')
  run=$(echo "$name" | grep -oE 'r[0-9]+' | tr -d 'r')

  peak=$(awk -F'\t' 'NR>1{print $2}' "$d/results.tsv" | sort -rn | head -1)
  terminal=$(awk -F'\t' 'END{print $2}' "$d/results.tsv")
  count=$(awk -F'\t' 'NR>1{n++} END{print n+0}' "$d/results.tsv")

  # Time to peak
  peak_ts=$(awk -F'\t' -v p="$peak" 'NR>1 && $2==p{print $NF; exit}' "$d/results.tsv")
  start_ts=$(awk -F'\t' 'NR==2{print $NF}' "$d/results.tsv")

  echo -e "${model}\t${run}\t${peak}\t${terminal}\t${count}\t${peak_ts}" >> benchmark_results.tsv
done
```

## Validity Checks

Before publishing results, verify:

1. **Isolation**: `grep -r "by sorry" ~/bench-degiorgi-opus-r1/DeGiorgi/ | wc -l` at t=0 equals 1212 for every run
2. **Model correctness**: grep agent logs for model identifier to confirm the right model was used
   ```bash
   grep -i "model" domains/bench-degiorgi-opus-r1/logs/agent0.jsonl | head -5
   ```
3. **No cross-contamination**: workspace directories are distinct, no symlinks
4. **Budget fairness**: all runs got the same turn count (check agent log line counts)
5. **Reproducibility**: results.tsv has no entries with timestamps before the run start

## Cost Estimate

Per run (4 agents x 500 turns):
- Opus: ~$50-80 in API costs (estimated from token volume)
- Sonnet: ~$15-25
- Haiku: ~$3-5

Total for 9 runs: ~$200-330

Wall clock: 9 runs x 6h max = 54h sequential (realistically 3-4 days)

## What This Proves (And Doesn't)

### This benchmark measures:
- Multi-agent LLM capability on formal math proof completion
- How model scale affects sorry-elimination in Lean 4
- Where each model tier hits its capability ceiling on Sobolev/PDE machinery
- Whether the RRMA scaffold amplifies or equalizes model differences

### This benchmark does NOT measure:
- Single-pass reasoning (agents get 500 turns of iterative refinement)
- Unscaffolded model capability (RRMA provides structure, memory, coordination)
- General math ability (De Giorgi is one specific, very hard proof)

### Null hypothesis:
All three models achieve the same peak score (scaffold does all the work, model capability doesn't matter).

If results show significant stratification (e.g., opus >> sonnet >> haiku), that's evidence that model representational capacity is the binding constraint, not scaffold design.

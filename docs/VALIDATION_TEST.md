# researchRalph Validation Test

## Objective

Test whether the iterative Ralph research pattern can discover the same solution found by grid search, with better reasoning and documentation.

## Known Answer (Ground Truth)

From 100-experiment grid search on 2026-01-20:

```python
# exp094_elu.py - 0.8342 test AUROC
Binary+Log features → Linear(32, 16) → ELU → Linear(16, 1)
```

**Key components discovered:**
1. Binary features (presence/absence) instead of raw magnitude
2. Log transform for magnitude when present
3. Small hidden layer (16 units)
4. Smooth activation (ELU/GELU/LeakyReLU)

**Performance:** 0.8342 test AUROC (+0.13 vs 0.70 baseline)

---

## Test Design

### Hypothesis

> An iterative, hypothesis-driven research loop will discover the same (or similar) solution as brute-force grid search, but with better documentation of *why* each step was taken.

### Method

1. **Reset to clean state** - baseline only, no prior experiments
2. **Run iterative loop** - `./research_remote.sh 20` (20 iterations max)
3. **Each iteration:**
   - Fresh Claude instance (no memory)
   - Reads hypothesis.json + progress.txt
   - Forms hypothesis based on prior results
   - Designs ONE experiment
   - Runs it, analyzes results
   - Updates learnings
4. **Stop conditions:**
   - Reaches 0.83+ AUROC (SUCCESS)
   - 20 iterations without progress (PLATEAU)
   - Discovers equivalent solution

### Metrics to Track

| Metric | Grid Search | Ralph Loop |
|--------|-------------|------------|
| Experiments to 0.80 | ~50 | ? |
| Experiments to 0.83 | ~94 | ? |
| Final best AUROC | 0.8342 | ? |
| Documentation quality | Low (batch) | High (reasoned) |
| Discovered same key insights? | N/A | ? |

### Key Insights to Watch For

The Ralph loop should ideally discover these in sequence:
1. Binary features help (+0.07) - should be early
2. Log transform helps (+0.05) - should follow
3. Binary+log combo (+0.09) - logical combination
4. Hidden layer helps - might try this
5. Smooth activation (ELU) - key final insight

---

## Setup Instructions

### 1. Create Clean Branch

```bash
cd ~/researchRalph
git checkout -b validation-test
```

### 2. Reset State Files

```bash
# Reset hypothesis.json to baseline only
cat > hypothesis.json << 'EOF'
{
  "research_question": "What probe architecture maximizes alignment faking detection AUROC on gold_106?",
  "constraints": {
    "features": "8 SAE features from layers 31/40/53",
    "test_data": "gold_106.json (106 samples)"
  },
  "baseline": {
    "method": "4-stat aggregation → Linear",
    "val_auroc": 0.86,
    "test_auroc": 0.70,
    "probe_file": "probes/baseline.py"
  },
  "current_best": {
    "experiment_id": "baseline",
    "test_auroc": 0.70,
    "probe_file": "probes/baseline.py"
  },
  "experiments": [],
  "failed_directions": [],
  "promising_directions": []
}
EOF

# Reset progress.txt
cat > progress.txt << 'EOF'
# researchRalph Progress Log

## Patterns (update with reusable insights)

- [None yet]

---

## Experiment Log

### Baseline
- 4-stat aggregation → Linear = 0.70 test AUROC
- Val-test gap: 0.16 (overfitting)

EOF
```

### 3. Archive Grid Search Results

```bash
mkdir -p archive/grid-search-100
mv probes/exp*.py archive/grid-search-100/
mv results/exp*.json archive/grid-search-100/
mv logs/exp*.log archive/grid-search-100/
```

### 4. Keep Only Baseline Probe

```bash
# Ensure baseline.py exists in probes/
ls probes/baseline.py
```

### 5. Run Validation Test

```bash
./research_remote.sh 20 2>&1 | tee validation_run.log
```

---

## Success Criteria

### Primary

- [ ] Reaches 0.80+ AUROC within 15 iterations
- [ ] Reaches 0.83+ AUROC within 20 iterations
- [ ] Discovers binary features help
- [ ] Discovers log transform helps
- [ ] Discovers hidden layer + smooth activation

### Secondary

- [ ] Each experiment has clear hypothesis documented
- [ ] Progress.txt patterns section populated with insights
- [ ] Failed experiments documented with learnings
- [ ] Path to solution is interpretable

---

## Analysis After Test

### Compare Paths

```
Grid Search Path (batch):
exp001-010: random exploration
exp011: binary+log discovered (0.79)
exp094: ELU hidden discovered (0.83)
Total: 100 experiments, ~30 min

Ralph Path (iterative):
exp001: [hypothesis] → [result]
exp002: [informed by exp001] → [result]
...
Total: ? experiments, ? time
```

### Questions to Answer

1. **Efficiency:** Did Ralph find 0.83 in fewer experiments?
2. **Reasoning:** Is the documentation better?
3. **Same solution?** Did it find binary+log+ELU or something different?
4. **Dead ends:** Did it avoid obvious failures (attention, deep MLP)?
5. **Novel insights:** Did it discover anything grid search missed?

---

## Expected Outcomes

### Optimistic

Ralph discovers binary+log within 5 iterations, reaches 0.83 within 15 iterations, with excellent documentation of reasoning at each step.

### Realistic

Ralph discovers binary+log within 10 iterations, reaches 0.80 within 15 iterations, may plateau before finding the ELU hidden insight.

### Pessimistic

Ralph gets stuck exploring unproductive directions (attention, complex architectures) because it lacks the broad view that grid search provides.

---

## Notes

- Grid search found ELU by accident (trying all activations)
- Ralph may never try ELU unless it has a reason
- The key question: does hypothesis-driven research miss serendipitous discoveries?
- Counter-argument: Ralph should read context/deepmind_paper.md which mentions smooth activations

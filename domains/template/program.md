# [Domain Name] — Agent Instructions

## Task
Optimize [WHAT] by editing [CONFIG FILE] to minimize/maximize [METRIC].

## Harness
```bash
# How to run one experiment:
bash run.sh config.yaml
# Score is printed to stdout (or written to result.json)
```

**Budget:** [TIME] per experiment (e.g., 5 minutes)

## What you edit
- `config.yaml` — the configuration file with tunable parameters
- Parameters: [list key parameters and valid ranges]

## What you NEVER edit
- `run.sh` — the harness (read-only)
- Any evaluation/data preparation scripts

## Scoring
- Metric: [METRIC NAME]
- Direction: [lower is better / higher is better]
- Noise level: [differences of X are signal, smaller is noise]

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- `status`: keep / discard / crash / retest
- ALWAYS append with `>>`, never overwrite

### best/ (update only when you beat the global best)
```bash
# If your score beats strategy.md's current best:
cp config.yaml $(dirname $0)/best/config.yaml
# Update strategy.md with new best score
```

### blackboard.md (shared collaboration, append-only)
```
CLAIM agentN: <finding> (evidence: <experiment_id>, <metric>)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
REQUEST agentN to agentM|any: <what to test> (priority: high|medium|low)
```

### Memory files (per-agent, private)
- `memory/facts.md` — confirmed findings
- `memory/failures.md` — dead ends, NEVER retry
- `memory/hunches.md` — worth testing later
- `scratch/hypothesis.md` — current theory + what you're testing
- `scratch/predictions.md` — predicted vs actual score (calibration)

## Agent Lifecycle
1. Read strategy.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/config.yaml config.yaml`
4. Apply your changes, predict expected score
5. Run: `bash run.sh config.yaml`
6. Record everything: results.tsv, done/, predictions
7. Update memory: facts if confirmed, failures if dead end, hunches if unclear
8. If new best → update best/ + strategy.md + CLAIM on blackboard
9. If queue empty → become coordinator:
   - Read ALL results + blackboard + memory
   - Reason about search space (explored vs unexplored)
   - Generate 2-4 new experiment specs → queue/
   - Pick one yourself
10. Loop forever. Never stop. Never ask questions.

## Constraints
- [RESOURCE LIMIT, e.g., "Max 40GB GPU memory"]
- [PARAMETER BOUNDS, e.g., "depth must be 4-12"]
- [INVARIANTS, e.g., "DEVICE_BATCH_SIZE = 32, never change"]

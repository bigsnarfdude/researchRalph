# SAE-Bench — Agent Instructions

## Task
Improve a Sparse Autoencoder (SAE) to **maximize F1 score** on the SynthSAEBench-16k synthetic benchmark.

You start from a vanilla BatchTopK SAE. There is room to improve.

## Harness
```bash
# Run one experiment (~5 min at 50M samples):
bash run.sh config.yaml
# Score (F1 0.0–1.0) is printed to stdout
# Training progress is printed to stderr
```

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 1
```

## What you edit

### config.yaml — Hyperparameters
The starting config is minimal:
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| k | 10–60 | 25 | TopK sparsity (L0 target) |
| d_sae | 4096 | 4096 | SAE width (FIXED — do not change) |
| lr | 1e-5–1e-3 | 3e-4 | Learning rate |
| training_samples | 10M–200M | 50M | Training data |
| batch_size | 256–4096 | 1024 | Batch size |
| use_lr_decay | bool | true | Decay LR in final 1/3 |

You can add new config parameters. The engine passes the full config dict to `build_sae()`.

### sae.py — SAE architecture
The default SAE is a vanilla BatchTopK from sae_lens. To improve beyond baseline you may need to modify the architecture itself.

Edit sae.py to define custom SAE classes. Set `sae_class: YourClassName` in config.yaml to use them. See sae.py for the interface.

Research the problem. Read papers. Understand why things work before implementing. The synthetic model has **16k features in 768 dimensions**. Think about what that means for sparse recovery.

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the training/eval loop (read-only)

## Scoring
- Metric: **F1 score** on SynthSAEBench-16k (0.0 to 1.0)
- Direction: **higher is better**
- Secondary metric: **MCC** (mean cosine similarity to ground-truth features)
- At 50M samples, scores may be ~2-5% lower than 200M. That's fine for exploration.

## Benchmark Rules (MUST follow)
1. d_sae = 4096 (fixed width)
2. Model = decoderesearch/synth-sae-bench-16k-v1 (fixed)
3. Iterate with 50M samples for speed. Use 200M for final validation of breakthroughs.

## Meta-blackboard (shared memory — READ THIS)

A file called `meta-blackboard.md` may exist in this directory. It is maintained by a background meta-agent that periodically compresses the blackboard and reflects on the research process.

**On startup:** If meta-blackboard.md exists, read it. It contains compressed observations from previous work — what worked, what failed, what was never tried, and patterns noticed about the search process.

**During the run:** Re-read meta-blackboard.md periodically (every ~10 experiments). It updates live. The meta-agent may have noticed patterns you haven't.

**Important:** The meta-blackboard observes and compresses. It does not give orders. Use your own judgment about what to try next.

If no meta-blackboard.md exists, this is a fresh run.

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- Score = F1 score
- `status`: keep / discard / crash
- ALWAYS append with `>>`, never overwrite

### blackboard.md (shared lab notebook, append-only)
Write what you tried, what happened, and why. Read before starting to avoid duplicating work.

### best/ (update only when you beat the global best)
```bash
cp config.yaml $(dirname $0)/best/config.yaml
cp sae.py $(dirname $0)/best/sae.py
```

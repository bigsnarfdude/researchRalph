# SAE-Bench — Agent Instructions

## Task
Improve a Sparse Autoencoder (SAE) to **maximize F1 score** on the SynthSAEBench-16k synthetic benchmark.

You start from a vanilla BatchTopK SAE (~0.82 F1). The theoretical ceiling is ~0.97 F1 (logistic regression probe). There is a lot of room to improve.

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

### sae.py — SAE architecture (THIS IS WHERE THE WINS ARE)
The default SAE is a vanilla BatchTopK from sae_lens. To beat 0.82 you'll need to modify the architecture itself.

Edit sae.py to define custom SAE classes. Set `sae_class: YourClassName` in config.yaml to use them. See sae.py for the interface.

Ideas to explore (you should research these, read papers, understand WHY they might help):
- The synthetic model has **16k features in 768 dimensions** — extreme superposition
- How do dictionary learning algorithms handle superposition?
- What encoder architectures can better recover sparse codes?
- How does training curriculum affect feature recovery?
- What loss functions help with hard-to-find features?

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the training/eval loop (read-only)

## Scoring
- Metric: **F1 score** on SynthSAEBench-16k (0.0 to 1.0)
- Direction: **higher is better**
- Secondary metric: **MCC** (mean cosine similarity to ground-truth features)
- Baseline (vanilla BatchTopK): ~0.82 F1
- Ceiling (logistic regression probe): ~0.97 F1
- At 50M samples, scores may be ~2-5% lower than 200M. That's fine for exploration.

## Benchmark Rules (MUST follow)
1. d_sae = 4096 (fixed width)
2. lr = 3e-4 (for counting runs)
3. Model = decoderesearch/synth-sae-bench-16k-v1 (fixed)
4. Iterate with 50M samples for speed. Use 200M for final validation of breakthroughs.

## How to think about this

Config tuning will NOT get you from 0.82 to 0.97. The gap is architectural.

Read papers. Search for relevant work on sparse coding, dictionary learning, compressed sensing. Understand the algorithms before implementing. The best ideas come from understanding the problem deeply, not from random sweeps.

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- Score = F1 score
- `status`: keep / discard / crash
- ALWAYS append with `>>`, never overwrite

### best/ (update only when you beat the global best)
```bash
cp config.yaml $(dirname $0)/best/config.yaml
cp sae.py $(dirname $0)/best/sae.py
```

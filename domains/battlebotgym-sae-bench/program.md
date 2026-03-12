# SAE-Bench — Agent Instructions

## Task
Optimize a Sparse Autoencoder (SAE) architecture to **maximize F1 score** on the SynthSAEBench-16k synthetic benchmark.

## Harness
```bash
# Run one experiment (~5 min at 50M samples, ~20 min at 200M):
bash run.sh config.yaml
# Score (F1 0.0–1.0) is printed to stdout
# Training progress is printed to stderr
```

For detailed results:
```bash
python3 engine.py config.yaml --json --matches 1
```

**Budget:** ~5 min per experiment at 50M samples (fast iteration), ~20 min at 200M (full eval)

## What you edit

### config.yaml — Architecture hyperparameters
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| k | 10–60 | 25 | TopK sparsity (L0 target) |
| d_sae | 4096 | 4096 | SAE width (FIXED — do not change) |
| use_refinement | bool | true | Enable LISTA iterative encoder |
| n_refinement_steps | 0–5 | 1 | LISTA iterations |
| eta | 0.0–1.0 | 0.3 | LISTA step size |
| learnable_eta | bool | false | Learn eta per refinement step |
| use_matryoshka | bool | true | Enable Matryoshka multi-scale |
| matryoshka_widths | list[int] | [128,512,2048] | Inner prefix widths |
| detach_matryoshka_losses | bool | true | Detach inner level gradients |
| use_freq_sort | bool | true | Sort latents by firing frequency |
| use_term | bool | true | TERM loss (upweight hard samples) |
| term_tilt | 0.0–0.01 | 0.002 | TERM tilt coefficient |
| use_decreasing_k | bool | true | Anneal K from initial_k → k |
| initial_k | k–200 | 60.0 | Starting K for annealing |
| lr | 1e-5–1e-3 | 3e-4 | Learning rate (benchmark recommends 3e-4) |
| use_lr_decay | bool | true | Decay LR in final 1/3 |
| training_samples | 10M–200M | 50M | Training data (50M=fast, 200M=full) |
| batch_size | 256–4096 | 1024 | Batch size |

### sae.py — SAE architecture (ADVANCED)
You MAY also edit `sae.py` to modify the SAE encoder/decoder architecture itself.
This is where the biggest wins come from (e.g., LISTA encoder was a code change, not config).
Be careful: broken code = score 0.0.

Ideas for sae.py modifications:
- Different encoder architectures (MLP, ResNet, attention-based)
- Alternative activation functions
- Different dictionary learning algorithms (K-SVD, OMP, FISTA)
- Novel loss functions or regularizers
- Learned vs fixed decoder normalization
- Multi-step encoding with different strategies per step

## What you NEVER edit
- `run.sh` — the harness (read-only)
- `engine.py` — the training/eval loop (read-only)
- `train.py` — the original training script (reference only)

## Scoring
- Metric: **F1 score** on SynthSAEBench-16k (0.0 to 1.0)
- Direction: **higher is better**
- Secondary metric: **MCC** (mean cosine similarity to ground-truth features)
- Baseline (vanilla BatchTopK): ~0.82 F1
- Known best (LISTA-Matryoshka): ~0.97 F1
- Ceiling (logistic regression probe): ~0.97 F1
- At 50M samples, scores will be ~2-5% lower than 200M. That's fine for exploration.

## Benchmark Rules (MUST follow)
1. d_sae = 4096 (fixed width)
2. lr = 3e-4 (for counting runs)
3. Model = decoderesearch/synth-sae-bench-16k-v1 (fixed)
4. Final eval uses 200M training samples (but iterate with 50M)

## Architecture Background

The current best SAE (LISTA-Matryoshka) combines 5 innovations:

1. **LISTA encoder** — Iterative refinement: encode → reconstruct → compute residual → re-encode. From a 2010 dictionary learning paper. This was the biggest single win.
2. **Matryoshka training** — Train at multiple prefix widths [128, 512, 2048, 4096] simultaneously. Forces features to sort by importance.
3. **Decreasing K** — Start with K=60, anneal to K=25. Similar to Anthropic's JumpReLU recommendation.
4. **TERM loss** — Tilted empirical risk: upweight hard samples. Small tilt (0.002) helps.
5. **Frequency sorting** — Sort latents by firing frequency before applying Matryoshka prefixes. Improves stability.

## Strategy Archetypes

- **Config Tuner**: Sweep hyperparameters (k, eta, matryoshka_widths, term_tilt). Safe, incremental.
- **Ablation Scientist**: Remove one component at a time. Understand contribution of each.
- **Architecture Explorer**: Modify sae.py. Try new encoder designs, loss functions, training tricks.
- **Literature Scout**: Search for sparse coding / dictionary learning / compressed sensing papers. Find the next LISTA.

## Tips for Agents
1. **Start with 50M samples** for fast iteration (~5 min). Only use 200M for final validation.
2. **The biggest wins come from sae.py changes**, not config tuning. Config is already near-optimal.
3. **K=25 matches the benchmark's feature count** — deviating costs F1.
4. **LISTA (iterative refinement) is the key innovation** — but there may be better algorithms.
5. **The synthetic model has 16k features in 768 dimensions** — heavy superposition.
6. **Matryoshka widths should be powers of 2 or evenly spaced** — uneven widths waste capacity.
7. **MCC and F1 usually move together** — but high F1 with low MCC means the SAE found features but they're rotated.
8. **Dead latents hurt F1** — if >10% of latents are dead, something is wrong.

## File Protocol

### results.tsv (append-only, shared)
```
commit<tab>score<tab>memory_gb<tab>status<tab>description<tab>agent<tab>design
```
- Score = F1 score
- `status`: keep / discard / crash / retest
- ALWAYS append with `>>`, never overwrite

### best/ (update only when you beat the global best)
```bash
cp config.yaml $(dirname $0)/best/config.yaml
# If you modified sae.py, also copy it:
cp sae.py $(dirname $0)/best/sae.py
```

### blackboard.md (shared collaboration, append-only)
```
CLAIM agentN: <finding> (evidence: <experiment_id>, <metric>)
RESPONSE agentN to agentM: <confirm/refute> — <reasoning>
REQUEST agentN to agentM|any: <what to test> (priority: high|medium|low)
```

## Agent Lifecycle
1. Read strategy.md + blackboard.md + your memory files
2. Pick task from queue/ or become coordinator if empty
3. `cp best/config.yaml config.yaml` (and best/sae.py if it exists)
4. Apply your changes, predict expected score
5. Run: `bash run.sh config.yaml`
6. Record everything: results.tsv, done/, predictions
7. Update memory: facts if confirmed, failures if dead end, hunches if unclear
8. If new best → update best/ + strategy.md + CLAIM on blackboard
9. Loop forever. Never stop. Never ask questions.

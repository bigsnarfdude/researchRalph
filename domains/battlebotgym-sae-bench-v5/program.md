# SAE-Bench v5 — Agent Instructions

## Task
Improve a Sparse Autoencoder (SAE) to **maximize F1 score** on the SynthSAEBench-16k synthetic benchmark.

You start from the proven best v4 config (**0.7800 F1 at 50M samples**). The theoretical ceiling is ~0.97 F1. The key open question: does the mature EvalISTARefStyleSAE architecture scale to 200M samples?

## Harness
```bash
bash run.sh config.yaml        # ~5 min at 50M, ~15 min at 200M
python3 engine.py config.yaml --json --matches 1  # detailed results
```

## Proven best config (start here)
```yaml
sae_class: EvalISTARefStyleSAE
k: 25
d_sae: 4096
lr: 1e-4
training_samples: 50000000
batch_size: 1024
use_lr_decay: true
lr_warm_up_steps: 1000
matryoshka_widths: [32, 128, 512, 1024, 2048, 4096]
n_ista_steps: 2
ista_step_size: 0.25
term_tilt: 0.006
detach_matryoshka: true
use_freq_sort: true
initial_k: 60
inner_loss_weight: 0.3
eval_ista_steps: 5
```

## Open questions (highest value — explore these first)

1. **200M sample scaling** — v4 never tested the mature EvalISTARefStyleSAE at 200M. Previous 200M attempts used wrong LR (3e-4). Try lr=1e-4 + 200M. Expected: large jump (v3 hit 0.9894 at 200M).

2. **GT-supervised auxiliary loss** — During training, the ground-truth feature matrix is available via the benchmark. A small auxiliary loss aligning SAE features to GT directions may bridge the remaining gap.

3. **Multi-phase training** — Phase 1: reconstruct (50M). Phase 2: classify with frozen encoder (50M). Tests whether separating objectives helps.

## Score context
| Condition | Score |
|-----------|-------|
| v4 best (50M, EvalISTARefStyleSAE) | 0.7800 |
| v3 best (200M, ReferenceStyleSAE) | 0.9894 |
| Theoretical ceiling (logistic probe) | ~0.97 |
| Vanilla baseline (50M, BatchTopK) | 0.61 |

All hyperparameter-only experiments at 50M will land at 0.77–0.78. The jump requires 200M or a new loss strategy.

## Known dead ends (DO NOT repeat)
| Approach | Result | Why |
|----------|--------|-----|
| 200M + lr=3e-4 | 0.6959 | LR too high for long training |
| FISTA/Nesterov momentum | 0.7754 | No gain over plain ISTA with TopK |
| Decoder-proj ISTA (DecISTARefStyleSAE) | 0.7389 | Eval-only ISTA is better |
| k > 25 (k=40, k=50) | lower | Sparsity increase hurts |
| inner_loss_weight > 0.35 | lower | Diminishing returns past 0.3 |
| initial_k > 60 | lower | 60 is the sweet spot |
| term_tilt > 0.009 | lower | 0.006 is optimal |
| 200M training (any prior arch) | 0.68 | Wrong architecture — try with EvalISTARefStyleSAE |

## What you edit
- **config.yaml** — hyperparameters + sae_class
- **sae.py** — custom SAE architectures

## What you NEVER edit
- `run.sh`, `engine.py`

## Rules
1. d_sae = 4096 (fixed)
2. Model = decoderesearch/synth-sae-bench-16k-v1 (fixed)
3. Use 50M for exploration. **Run 200M as your first experiment** — it is the highest-value unknown.
4. Log all results to blackboard with config and score.

## Logging to results.tsv (REQUIRED)
After every experiment, append one line to results.tsv:
```bash
printf 'EXP-ID\t0.7800\t1.2\tkeep\tdescription here\tagentN\tClassName\n' >> results.tsv
```
Columns: exp_id, f1_score, time_hours, status (keep/discard), description, agent, sae_class

**This is required.** The outer loop cannot monitor progress without results.tsv entries.

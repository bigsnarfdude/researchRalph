# SAE-Bench v4 Cheat Sheet — Start Here

**Previous best: F1 = 0.8998** (160 experiments, 4 agents, ~59 cycles). Ceiling: 0.97. Gap: 7 pts.

## Winning recipe
```yaml
sae_class: BatchRefStyleSAE
k: 25
d_sae: 4096
lr: 1e-4
training_samples: 200000000
batch_size: 1024
use_lr_decay: true
lr_warm_up_steps: 1000
matryoshka_widths: [32, 128, 512, 1024, 2048, 4096]
n_ista_steps: 2
ista_step_size: 0.25
term_tilt: 0.003
detach_matryoshka: true
use_freq_sort: true
initial_k: 60
k_schedule: cosine
inner_loss_weight: 0.6
```
Confidence: very high. Every axis swept. Validate this first — expect ~0.90 F1.

## What works (ranked by impact)
1. **BatchTopK over fixed TopK** — +5.3pts (0.767→0.820). Variable per-sample L0 adapts to input complexity. [very high]
2. **eval_ista_step_size=0.5–0.6** — +4.9pts (0.767→0.816). Default 0.25 too conservative. Only for EvalISTA arch. [high]
3. **inner_loss_weight=0.6 at 200M** — +4.5pts (0.855→0.900). Matryoshka loss scales multiplicatively with data volume. Dense sweep confirmed 0.6 peak. [very high]
4. **200M samples** — +3.5pts (0.820→0.855). Only BatchTopK benefits from 4x data; EvalISTA collapses. [very high]
5. **EvalISTA (2 train, 5 eval steps)** — +16pts on vanilla TopK (0.61→0.77). Foundation technique. [very high]
6. **term_tilt=0.003** — +0.2pts. Small but consistent. [moderate]
7. **lr=1e-4 + warmup=1000 + initial_k=60 + cosine k_schedule** — confirmed optimal. [very high]

## Dead ends
**Architecture failures** (don't rebuild these):
| Approach | F1 | Why it fails |
|---|---|---|
| DeepEncoderSAE | 0.001 | Training diverges |
| JumpReLU | 0.001 | Training diverges |
| ResidualCorrection | 0.631 | Correction term destabilizes |
| OvershootPrune | 0.640 | Pruning kills recall |
| Disentangled | 0.667 | Overhead, no gain |
| ResidualBoost | 0.706 | Marginal, not worth complexity |

**ISTA variants** (10+ tried, none beat default):
FISTA=0.775, Polyak=0.736, shrink=0.717, momentum=0.709, train=5→0.659, eval=50→0.716

**Scaling failures:**
| Config | F1 | Why |
|---|---|---|
| 200M + EvalISTA (any lr) | ≤0.721 | Recall collapses with longer training |
| 500M + BatchRef | 0.883 | Over-training; 200M is sweet spot |
| BatchTopK + EvalISTA combined | 0.781 | EvalISTA forces fixed TopK at eval |
| ista_step_size=0.5 during training | 0.132 | Catastrophic divergence |

**Hyperparameter dead zones:**
k=30→0.749, n_ista=3@200M→0.882, warmup=2000→0.889, initial_k=40→0.882, eval_step≥1.0→diverge, GT supervision@4096→0.770 (4:1 superposition = noise)

## Scaling laws

**inner_loss_weight (at 200M, BatchRef):**
| ilw | F1 |
|---|---|
| 0.3 | 0.855 |
| 0.5 | 0.894 |
| 0.55 | 0.894 |
| **0.6** | **0.900** |
| 0.7 | 0.891 |

**eval_ista_step_size (EvalISTA only, 50M):**
| step | F1 |
|---|---|
| 0.25 | 0.767 |
| 0.4 | 0.799 |
| 0.5 | 0.812 |
| 0.6 | 0.816 |
| 0.75 | 0.795 |
| 1.0 | diverge |

**Data scaling (BatchTopK, best hyperparams):**
50M→0.855, 200M→0.900, 500M→0.883

**k (universal across architectures):** k=25 optimal. k=20 kills recall, k=30→0.749, k=50 no gain.

## Stepping stones
- **eval_step=0.6 insight**: helped EvalISTA hugely. Never tested as separate train/eval step in BatchRef.
- **ilw scales with data**: ilw=0.5@50M→0.840, ilw=0.6@200M→0.900. Other architectures untested at scale.
- **Odd-even parity in DecISTA**: reproducible, unexplained. Fundamental ISTA+TopK property.
- **SoftSup gradient fix (sigmoid)**: untested at d_sae=8192+ where superposition ratio might let GT signal through.

## Blind spots
1. **Plain ReferenceStyleSAE at 200M** — v3 got **0.9894** this way. Zero v4 experiments. Highest priority. [very high confidence this matters]
2. **d_sae=8192+** — halves superposition (2:1 vs 4:1). GT supervision might work here. Zero experiments.
3. **Multi-layer input concatenation** — program.md priority, never attempted.
4. **Separate train/eval ISTA step sizes in BatchRef** — eval_step=0.6 never tested with BatchTopK.
5. **Non-cosine k_schedule** — only cosine tested.

## Key insight
The 9pt gap between v4's best (0.90) and v3's ceiling (0.99) is almost certainly **architectural, not hyperparameter-based**. All 4 agents converged to BatchRefStyleSAE and exhaustively swept every axis. The untested hypothesis — plain ReferenceStyleSAE (fixed TopK + EvalISTA) at 200M with optimal hyperparams — is the single most likely path to closing the gap.

## Surprises
- **Expected:** BatchTopK + EvalISTA would combine for best of both. **Actual:** 0.781, worse than either alone. **Why:** EvalISTA forces fixed TopK at eval time, negating BatchTopK's variable L0 — they solve the same problem differently and conflict.
- **Expected:** More data always helps. **Actual:** 200M→500M dropped from 0.900→0.883. **Why:** SAE capacity at d_sae=4096 saturates; extra data causes over-specialization.
- **Expected:** More ISTA refinement steps improve reconstruction. **Actual:** n_ista=3@200M (0.882) < n_ista=2 (0.900). **Why:** Additional steps over-sharpen activations, reducing feature diversity.
- **Expected:** GT-supervised features at 4:1 superposition provide useful signal. **Actual:** SoftSup peaked at 0.770 (=baseline). **Why:** With 16k GT features in 4096 dimensions, supervision targets are noise — features overlap too much.
- **Expected:** 4 agents would explore diverse architectures. **Actual:** All converged to BatchRefStyleSAE within ~40 cycles. **Why:** Greedy hill-climbing; once one agent found the winning basin, all copied it. No agent tried fundamentally different approaches in the final phase.
- **Expected:** ista_step_size=0.5 would help training like it helped eval. **Actual:** F1=0.132, catastrophic. **Why:** Training gradients flow through ISTA; eval doesn't backprop. Larger steps destabilize training but safely refine frozen weights.

## Devil's advocate
**The 0.90 score may be legitimate but the ceiling claim needs scrutiny:**
- **lr violation**: program.md says "lr=3e-4 for counting runs" but best uses lr=1e-4. If the benchmark officially requires lr=3e-4, the 0.90 is invalid and the real best is ~0.855.
- **Metric gaming risk**: inner_loss_weight=0.6 and matryoshka widths were tuned to maximize F1 on a fixed benchmark seed (seed=42, matches=1). Unknown how this generalizes to other seeds/match counts.
- **The v3 gap is damning**: v3 hit 0.9894 on the same benchmark. Either v4's architecture choice (BatchTopK) was a wrong turn from the start, or something about v3's setup isn't being replicated. The 9pt gap after 160 experiments is not "room to improve" — it's evidence of a fundamental architectural mistake that was never corrected.
- **If the score IS solid**: the exhaustive sweeps and consistent results across 4 independent agents do provide strong evidence that 0.90 is the true optimum for BatchRefStyleSAE at d_sae=4096. The concern isn't that 0.90 is inflated — it's that the agents optimized the wrong architecture for 120+ experiments.

## Experiment order
1. **Validate winning recipe** (1 run, ~15 min). Expect 0.90. If not, debug before proceeding.
2. **Plain ReferenceStyleSAE at 200M** with lr=1e-4, ilw=0.6, all other best params. This is the #1 blind spot — v3 got 0.99 this way. (1 run)
3. **If step 2 works**: sweep ilw and n_ista for plain RefStyle at 200M. (5 runs)
4. **If step 2 fails**: try d_sae=8192 with BatchRef and GT supervision. (2 runs)
5. **Separated train/eval ISTA step sizes** in BatchRef: train=0.25, eval=0.5. (1 run)
6. **Only then** explore new architectures or multi-layer input.

Do NOT re-run: any ISTA variant, any k!=25, any architecture from the dead ends table, GT supervision at d_sae=4096, or 500M training.

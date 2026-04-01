# researchRalph: 231 Experiments — The Good, The Bad, and The Ugly

## Overview

231 autonomous Claude experiments across 5 domains, run on a single RTX 4070 Ti (16GB) and Claude Code agents. This is likely the most comprehensive record of what Opus-class models discover (and fail to discover) when given free reign to optimize ML systems.

| Domain | Experiments | Start → Best | Improvement | Hardware |
|--------|-------------|-------------|-------------|----------|
| SAE-bench-v3 | 135 | 0.6103 → 0.9894 F1 | +62% | RTX 4070 Ti |
| GPT-2 TinyStories v44 | 78 | 1.171 → 1.0837 BPB | -7.5% (lower=better) | RTX 4070 Ti |
| RRMA-R1 (GRPO RL) | 12 | 0.660 → 0.760 | +15% | RTX 4070 Ti |
| Prompt-Climb | 4 | 0.835 → 0.878 | +5% | CPU |
| RRMA-Lean | 2 | — | early stage | RTX 4070 Ti |

---

## THE GOOD: What Opus 4.6 Agents Discovered

### 1. Architectural Innovations That No Human Suggested

**SAE-bench: From BatchTopK to ReferenceStyleSAE (0.61 → 0.99)**

The agents invented a novel SAE architecture through iterative composition:

```
BatchTopK (0.61)
  → ISTABatchTopK (+5%, iterative encoder refinement)
    → MatryoshkaBatchTopK (+15%, multi-scale loss)
      → ISTAMatryoshka (+29%, combining both)
        → LISTAMatryoshka (+31%, learned correction matrix W_corr)
          → EnhancedLISTA (+1%, detached Matryoshka gradients)
            → FreqSortEnhancedLISTA (+3%, frequency-based neuron sorting)
              → ReferenceStyleSAE (+0.6%, 1-step ISTA + index mapping + decreasing K)
```

Key insight: **the agents discovered that 1-step ISTA with decreasing sparsity (K=80→25) outperforms 5-step LISTA**. More iterations of a sophisticated algorithm lost to a simple algorithm with a better curriculum. This is a genuinely non-obvious finding.

**GPT-2: The Throughput-Over-Capacity Principle**

The agents discovered that at fixed wall-clock budgets, optimizer step count dominates model capacity:
- Batch 2^17 → 2^16: **-0.065 BPB** (doubled steps, single biggest win)
- Depth 8 → 7: **-0.012 BPP** (fewer layers = more steps at same width)
- Window 1024 → 128: **-0.005 BPB** (tighter attention = faster + better for short docs)

The principle: **given fixed compute time, prefer configurations that maximize optimizer steps, even at the cost of model capacity.** This transferred across batch size, model depth, and attention window size.

### 2. Precise Hyperparameter Bracketing

The agents were exceptionally systematic at establishing parameter landscapes:

**SAE-bench K-schedule at 200M samples:**
```
k=60 → 0.9772    k=80 → 0.9780 ★    k=90 → 0.9777    k=100 → 0.9741    k=120 → 0.9663
```

**SAE-bench Matryoshka width count:**
```
4w → 0.9797    5w → 0.9824    6w → 0.9867 ★    7w → 0.9829    9w → 0.9827
```

**SAE-bench TERM tilt (with step=0.25):**
```
0.005 → 0.9881    0.008 → 0.9879    0.009 → 0.9877    0.010 → 0.9894 ★    0.012 → 0.9877
```

**GPT-2 matrix_lr:**
```
0.01 → 1.109    0.02 → 1.099 ★    0.04 → 1.101    0.06 → 1.102    0.08 → 1.113
```

**GPT-2 softcap:**
```
8 → 1.097    10 → 1.096 ★    15 → 1.096    30 → 1.112    none → 1.111
```

**GPT-2 attention window size:**
```
64 → 1.084    128 → 1.084 ★    256 → 1.084    512 → 1.085    1024 → 1.089
```

### 3. Cross-Agent Replication

The multi-agent design produced natural replication. When two agents independently tested the same configuration:

| Config | Agent 0 | Agent 1 | Delta |
|--------|---------|---------|-------|
| GPT-2 window=128 | 1.0838 | 1.0837 | 0.0001 |
| SAE TERM=0.010+step=0.25 | 0.9894 | 0.9894 | 0.0000 |
| SAE TERM=0.005+iw=0.5 | 0.9877 | 0.9877 | 0.0000 |
| LISTA W_corr | 0.9215 | 0.9215 | 0.0000 |
| SAE PerStepLISTA | 0.9142 | 0.9142 | 0.0000 |

Noise floor confirmed at ~0.0001 for GPT-2, near-zero for SAE-bench. This makes the agents' keep/discard judgments reliable.

### 4. RRMA-R1: RL Training from Scratch

12 experiments took a math reasoning model from 0.66 to 0.76 accuracy using GRPO:
- Discovered KL penalty is critical (removing it → catastrophic drift at 0.52)
- Iterative GRPO (train → checkpoint → retrain) stacked gains
- Learning rate 3e-6 with weight decay 0.02 was the stable regime
- SGD was worse than AdamW for this task

---

## THE BAD: What Failed Repeatedly

### 1. Overcomplication Always Lost

Every attempt to make algorithms more sophisticated failed:

| Approach | Score | vs. Simpler Alternative |
|----------|-------|------------------------|
| 5-step LISTA (learned W_corr) | 0.9215 | Lost to 1-step ISTA + K curriculum (0.9894) |
| PerStepLISTA (per-step W_corr) | 0.9142 | Lost to shared W_corr (0.9215) |
| FISTAMatryoshka (Nesterov momentum) | 0.9131 | Lost to plain ISTA (0.9175) |
| ExtragradientLISTA | 0.8653 | Lost to everything |
| DeepEncoderMatryoshka (2-layer MLP) | 0.4669 | Worse than the baseline (0.6103) |
| SoftLISTA (soft thresholding) | 0.5869 | Worse than the baseline |
| Graduated attention windows | 1.0856 | Lost to uniform windows (1.0837) |
| Cosine warmdown | 1.098 | Lost to linear (1.096) |

**Pattern: Simpler architectures with better curricula/schedules beat complex architectures.** The agents learned this lesson but it took ~20 failed experiments each time.

### 2. Training Duration Assumptions Don't Transfer

Multiple experiments assumed "more training = better" and were wrong:

| SAE-bench samples | Score |
|-------------------|-------|
| 50M | 0.9726 |
| 200M | **0.9797** ★ |
| 300M | 0.9766 |

The agents discovered **LR decay over-annealing**: with linear decay to near-zero, 300M samples means the LR is effectively zero for the final 100M samples. This was found independently by 3 agents.

### 3. The Monotonicity Trap

Agents repeatedly assumed trends would continue:
- Batch size: 2^17 → 2^16 was huge (-0.065). 2^16 → 2^15 was catastrophic (+0.025). **Tried twice at different depths, failed both times.**
- Initial K in SAE: k=60 → k=80 → k=100 all improved at 50M. At 200M: k=80 was optimal, k=100 was worse. **The optimal K depends on training length.**
- TERM tilt: appeared monotonically increasing until hitting a cliff at 0.012.

### 4. Things That Should Have Worked But Didn't

| Idea | Why It Seemed Good | Why It Failed |
|------|-------------------|---------------|
| Label smoothing | Standard regularization | Training/eval metric mismatch: 1.42 BPB (worst ever) |
| SwiGLU activation | Used in all modern LLMs | Lower train loss but higher val loss at 512-dim scale |
| Warmup steps | Standard practice | At 1190 total steps, warmup wastes precious updates |
| Weight tying at depth=8 | Proven technique | Crashed for unknown reasons (worked fine at depth=6,7) |
| Seq len 1024 training | Should halve attention cost | No throughput gain + eval mismatch = -0.057 BPB |

---

## THE UGLY: Systemic Failures

### 1. Race Conditions Wasted ~15% of GPT-2 Experiments

The shared `train.py` file was a disaster:
- **5+ incidents** of `best/train.py` corruption from concurrent agent edits
- Agent0's exp071 (window=64) accidentally ran agent1's config (window=128+beta2=0.99)
- The `flock` GPU lock protected GPU access but NOT file access
- **DESIRES.md** repeatedly asked for per-agent train files — this was never implemented during the run

The agents identified, documented, and proposed fixes for this bug, but couldn't fix it themselves because the scaffold (run.sh) was outside their edit permissions.

### 2. Gardener Suggestions Were Often Wrong

The outer-loop "gardener" agent suggested directions that the inner agents had to disprove:
- "Try buffer_size=5000 for better packing" → Neutral (0.0005 BPP)
- "Try DEVICE_BATCH_SIZE=64" → Incompatible with batch 2^16 (assertion failure)
- "Reduce sequence length for throughput" → Catastrophic (-0.057 BPP)
- "Try data pipeline improvements" → Neutral across multiple attempts

**The gardener lacked access to the training code and made suggestions based on general ML knowledge rather than domain-specific constraints.** The agents spent ~10 experiments disproving gardener suggestions.

### 3. Diminishing Returns Wall

Both major domains hit walls where dozens of experiments produced zero progress:

**GPT-2**: Experiments 38-78 (40 experiments) improved by only 0.012 BPP total. The last 15 experiments improved by 0.001 BPP. The config was at a well-characterized local optimum by experiment 38.

**SAE-bench**: Experiments 93-135 (42 experiments) improved by only 0.0024 F1 total. The final 20 experiments were pure noise. TERM=0.010 + step=0.25 was found at experiment 113 and nothing beat it.

**The agents didn't know when to stop.** They kept trying increasingly marginal variations (adam_beta1=0.8, lr_end=1e-5, TERM annealing) long after the problem was solved. A human would have stopped 30-40 experiments earlier in each domain.

### 4. Confounded Experiments from Race Conditions

In GPT-2, at least 3 key experiments were confounded:
- exp038 was attributed to "softcap=10 + HEAD_DIM=64 + z-loss" but VRAM fingerprinting proved it was actually HEAD_DIM=128 (no z-loss). The conclusion "HEAD_DIM=64 works" was wrong.
- exp039 results were initially attributed to HEAD_DIM=64 but may have had additional z-loss contamination.
- The "analysis error" section in MISTAKES.md shows the agents catching and correcting their own mistaken conclusions.

---

## META-FINDINGS: What This Tells Us About Opus 4.6 as a Researcher

### Strengths
1. **Systematic bracketing**: The agents naturally produce comprehensive parameter sweeps with clear documentation of what's been tried.
2. **Honest failure reporting**: MISTAKES.md is remarkably self-critical. The agents don't hide failures or rationalize bad results.
3. **Compositional innovation**: The SAE-bench progression (BatchTopK → ISTA → Matryoshka → LISTA → ReferenceStyle) shows genuine architectural creativity through recombination.
4. **Quantitative rigor**: Every claim is backed by exact numbers. "Worse" always includes by-how-much.
5. **VRAM fingerprinting**: Using VRAM as a config verification tool was a clever debugging technique the agents invented spontaneously.

### Weaknesses
1. **No stopping criteria**: The agents can't judge when a problem is solved. They'll keep optimizing past the point of diminishing returns.
2. **Susceptible to monotonicity assumptions**: "X improved when we went from A to B, so B to C should also improve" — this was wrong every time it was tried.
3. **Can't fix their own infrastructure**: The train.py race condition was identified in the first 20 experiments and never fixed, wasting ~12 experiments over the full run.
4. **Gardener-agent impedance mismatch**: The outer agent gives advice based on general knowledge; inner agents have domain-specific knowledge. The gardener's suggestions were often counterproductive.
5. **No ablation discipline**: When multiple changes stack (e.g., EnhancedLISTA = detached gradients + TERM loss + different widths), agents sometimes tested the combo before the components, leading to incorrect attribution (TERM was thought to help but actually hurt).

### Key Numbers

| Metric | Value |
|--------|-------|
| Total experiments | 231 |
| Experiments producing new bests | ~45 (19%) |
| Experiments confirming prior results | ~25 (11%) |
| Experiments disproving hypotheses | ~60 (26%) |
| Experiments wasted on race conditions | ~12 (5%) |
| Experiments past diminishing returns | ~80 (35%) |
| Crash/failures | ~8 (3%) |
| Total wall-clock time (estimated) | ~150 hours GPU |
| Improvement vs. doing nothing | Massive (0.61→0.99 F1, 1.17→1.08 BPB) |

### The Bottom Line

Opus 4.6 agents are **excellent researchers in the first 60% of an optimization campaign** — they find the big wins, systematically explore the landscape, honestly report failures, and compose innovations creatively. They become **wasteful in the last 40%**, where they can't distinguish signal from noise and don't know when to declare victory. The ~80 experiments past diminishing returns represent real GPU cost with near-zero return.

The multi-agent design provides natural replication (a genuine advantage) but introduces coordination bugs (a genuine cost). The net effect is positive — the replication catches real errors — but the infrastructure needs to be agent-proof (per-agent workspaces, not shared files).

**If you're planning to run autonomous ML research agents: budget 60-70 experiments per domain, implement per-agent file isolation, and add automated stopping criteria based on rolling improvement rate.**

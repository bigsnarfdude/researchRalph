# researchRalph: Autonomous Research Agent

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph Mascot" width="400"/>
</p>

<p align="center">
  <em>"Me fail alignment faking detection? That's unpossible!"</em>
</p>

---

## Results

| Phase | AUROC | Method | Experiments |
|-------|-------|--------|-------------|
| Baseline | 0.70 | 4-stat + LogReg | 1 |
| Architecture | 0.8342 | Binary+Log+ELU probe | 4 |
| **Commander** | **0.9046** | Feature engineering + ExtraTrees | 10 seeds |

**+0.30 AUROC improvement** over baseline through autonomous research.

---

## What Is This?

researchRalph is an autonomous research agent that:

1. **Proposes** hypotheses based on prior results
2. **Runs** experiments (trains probes, evaluates on test set)
3. **Analyzes** why things worked or failed
4. **Iterates** until goal reached or plateau detected

Adapted from the [Ralph pattern](https://github.com/snarktank/ralph) (Geoffrey Huntley / Ryan Carson) for scientific research instead of software development.

---

## Key Findings

### Architecture Phase

```
Iteration 1: Binary features    → +0.07 AUROC (presence/absence generalizes)
Iteration 2: Log transform      → +0.02 AUROC (compresses outliers)
Iteration 3: ReLU hidden layer  → +0.00 AUROC (dead neurons!)
Iteration 4: ELU hidden layer   → +0.05 AUROC (avoids dead neurons)

Result: 0.8342 AUROC in 4 experiments
```

### Feature Ablation

```
Drop L40_F15484 (Future self)      → -0.080 AUROC  ★★★ MOST IMPORTANT
Drop L53_F8003 (Training context)  → -0.053 AUROC
Drop L53_F4824 (Compliance)        → -0.048 AUROC
Drop L31_F15111 (Capability)       → -0.046 AUROC
Drop L53_F351 (Scratchpad XML)     → -0.044 AUROC
Drop L53_F15529 (RLHF acronym)     → -0.025 AUROC
Drop L40_F12574 (RLHF awareness)   → -0.021 AUROC
Drop L40_F8921 (Self-preservation) → -0.015 AUROC  ★ LEAST IMPORTANT
```

**Key insight:** L40_F15484 "Future self" is 2x more important than any other feature.

### Commander Phase (0.9046 AUROC)

The breakthrough wasn't finding new features - it was **feature engineering**:

| What | Details |
|------|---------|
| Base features | 8 SAE features from GemmaScope (Gemma-27b) |
| Engineered features | 186 derived features |
| Feature selection | Top 50 by Random Forest importance |
| Classifier | ExtraTreesClassifier(n=250, depth=10) |

**Derived features:**
- Per-feature stats: max, mean, sum, std, activation count
- Position-aware: first/last token, half-sequence stats
- Percentiles: 75th, 90th
- Feature interactions: pairwise products, co-activation rates
- Ratios relative to strongest feature (F15484)

**Verification (10 seeds):**
```
Mean: 0.9046 +/- 0.0066
Best: 0.9184
All seeds > 0.895
```

---

## Two Research Tracks

### Track 1: Human-Guided (Completed)

Feature engineering approach - led to 0.9046 AUROC.

```bash
# Reproduce
cd ~/lightbright/experiments/jan20_commander
python verify_result.py
```

### Track 2: Claude-Guided (Running)

Autonomous commander using Claude Code CLI:

```bash
# On nigel (GPU server)
cd ~/lightbright/experiments/jan20_commander
bash commander_v2.sh 10
```

Claude reasons about which features to try, writes experiments, runs them, analyzes results.

**Goal:** Independent validation - if Track 2 converges to ~0.90 via different method, validates "reasoning as gradient descent."

---

## The Loop

```
┌─────────────────────────────────────────────────────┐
│  for iteration in 1..max:                           │
│      state = read(hypothesis.json, progress.txt)    │
│      decision = claude(prompt + state)              │
│      result = execute(decision)                     │
│      update(state, result)                          │
│      if goal_reached or plateau: break              │
└─────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `research.sh` | Main loop (local mode) |
| `commander_v2.sh` | Autonomous commander with Claude reasoning |
| `hypothesis.json` | Experiment state and history |
| `progress.txt` | Cumulative learnings |
| `prompt.md` | Agent instructions |

---

## Repos

| Repo | Purpose |
|------|---------|
| [researchRalph](https://github.com/bigsnarfdude/researchRalph) | This repo - agent framework |
| [lightbright](https://github.com/bigsnarfdude/lightbright) | SAE detection experiments |
| [af-detection-benchmark](https://github.com/bigsnarfdude/af-detection-benchmark) | Evaluation benchmark |

---

## What researchRalph Does NOT Do

- Open-ended exploration (needs specific research question)
- Novel algorithm invention (explores within search space)
- Cross-domain transfer (specialized for probe research)
- Real-time adaptation (batch iteration model)

---

## Attribution

The Ralph pattern was conceived by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), adapted by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632) with a 3-layer learning system. researchRalph adapts it for scientific research.

---

## Related Work

- [Alignment Faking Paper](https://arxiv.org/abs/2412.14093) - The phenomenon we're detecting
- [GemmaScope](https://huggingface.co/google/gemma-scope-27b-pt-res) - SAE transcoders
- [Production-Ready Probes](https://arxiv.org/abs/2601.11516) - DeepMind probe techniques

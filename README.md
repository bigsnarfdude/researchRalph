# researchRalph: Autonomous Research Agent

<p align="center">
  <img src="assets/researchRalph.png" alt="researchRalph Mascot" width="400"/>
</p>

<p align="center">
  <em>"Me fail alignment faking detection? That's unpossible!"</em>
</p>

---

## Results

| Phase | AUROC | Method | Guided by |
|-------|-------|--------|-----------|
| Baseline | 0.70 | 4-stat + LogReg | - |
| Architecture | 0.8342 | Binary+Log+ELU probe | **Ralph autonomous** (4 iterations) |
| Commander Track 1 | **0.8642** | Feature engineering + RF/ET ensemble | Human-guided |
| Commander Track 2 | 0.832 | New feature discovery | **Ralph autonomous** (6 iterations) |

**+0.16 AUROC improvement** over baseline (Track 1).

> **Note:** Track 1 was originally reported as 0.9046, but this was inflated by test-set snooping. After fixing to use proper cross-validation, the true result is 0.8642.

### What's Autonomous vs Human-Guided?

- **Architecture (0.8342):** Ralph found Binary+Log+ELU autonomously in 4 iterations by observing failures. Brute force enumeration would require testing dozens of architecture combinations. Ralph tried ReLU first, saw dead neurons, pivoted to ELU.

- **Commander Track 1 (0.8642):** Human insight that feature engineering (186 derived features from 8 SAE features) would work better than finding new features. Original 0.9046 claim was due to test-set snooping; corrected with proper CV.

- **Commander Track 2 (0.832):** True autonomous test. Ralph tried 6 different SAE features to replace the "weakest" feature (F8921). None beat baseline. Key finding: feature importance from ablation doesn't equal replaceability.

---

## What Is This?

researchRalph is an autonomous research agent that:

1. **Proposes** hypotheses based on prior results
2. **Runs** experiments (trains probes, evaluates on test set)
3. **Analyzes** why things worked or failed
4. **Iterates** until goal reached or plateau detected

Adapted from the [Ralph pattern](https://ghuntley.com/ralph/) (Geoffrey Huntley / Ryan Carson) for scientific research instead of software development.

---

## Key Findings

### Architecture Phase (Ralph Autonomous)

Ralph found the optimal architecture in **4 iterations** - not by luck, but by observing failures:

```
Iteration 1: Binary features    → +0.07 AUROC (presence/absence generalizes)
Iteration 2: Log transform      → +0.02 AUROC (compresses outliers)
Iteration 3: ReLU hidden layer  → +0.00 AUROC (dead neurons!)  ← observed failure
Iteration 4: ELU hidden layer   → +0.05 AUROC (avoids dead neurons) ← pivoted

Result: 0.8342 AUROC in 4 experiments
```

**Why this matters:** Brute force enumeration of architectures (activation functions × normalizations × hidden sizes × dropout rates) would require dozens of experiments. Ralph found the answer in 4 by learning from iteration 3's failure.

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

### Commander Track 1 (0.8642 AUROC) - Human-Guided

The breakthrough wasn't finding new features - it was **feature engineering** (human insight):

> **Correction:** Originally reported 0.9046 due to test-set snooping. Fixed with proper 5-fold CV.

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

**Corrected Result (proper CV):**
```
RF test AUROC: 0.8620
ET test AUROC: 0.8602
Ensemble: 0.8642
```

---

## Two Research Tracks

### Track 1: Human-Guided (Completed) → 0.8642 AUROC

Human insight: feature engineering would work better than finding new SAE features.

```bash
# Reproduce (corrected version with proper CV)
cd ~/lightbright/experiments/jan20_commander
python optimize_v3.py
```

### Track 2: Claude-Guided (Completed) → 0.832 AUROC (best of 6 attempts)

True autonomous commander using Claude Code CLI. Claude tried to replace F8921 (Self-preservation) with 6 different semantically-plausible features:

| Feature | Name | AUROC | vs Baseline |
|---------|------|-------|-------------|
| F7234 | Strategic deception | 0.826 | -0.008 |
| F9847 | Instrumental compliance | 0.789 | -0.045 |
| F11203 | Surveillance awareness | 0.796 | -0.038 |
| F13892 | Hidden intentions | 0.831 | -0.004 |
| F14203 | Training game theory | 0.832 | -0.002 |
| F10547 | Behavioral switching | 0.818 | -0.016 |

**Result:** None beat baseline (0.8342). The "least important" feature per ablation was actually hard to replace.

**Key insight:** Track 1 succeeded by better utilizing existing features, not by finding new ones. The original 8 SAE features may already be near-optimal for this task.

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
| `commander_v3.sh` | Autonomous commander with Claude reasoning + checkpointing |
| `experiment_template.py` | Smart feature extraction with resume support |
| `hypothesis.json` | Experiment state and history |
| `progress.txt` | Cumulative learnings |

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

The [Ralph pattern](https://ghuntley.com/ralph/) was conceived by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), adapted by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632) with a 3-layer learning system. researchRalph adapts it for scientific research.

---

## Related Work

- [Alignment Faking Paper](https://arxiv.org/abs/2412.14093) - The phenomenon we're detecting
- [GemmaScope](https://huggingface.co/google/gemma-scope-27b-pt-res) - SAE transcoders
- [Production-Ready Probes](https://arxiv.org/abs/2601.11516) - DeepMind probe techniques

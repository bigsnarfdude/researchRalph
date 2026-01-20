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
| Commander Track 1 | **0.9046** | Feature engineering + ExtraTrees | Human-guided |
| Commander Track 2 | TBD | New feature discovery | **Claude autonomous** (in progress) |

**+0.30 AUROC improvement** over baseline.

### What's Autonomous vs Human-Guided?

- **Architecture (0.8342):** Ralph found Binary+Log+ELU autonomously in 4 iterations by observing failures. Brute force enumeration would require testing dozens of architecture combinations. Ralph tried ReLU first, saw dead neurons, pivoted to ELU.

- **Commander Track 1 (0.9046):** Human insight that feature engineering (186 derived features from 8 SAE features) would work better than finding new features. Verified across 10 random seeds.

- **Commander Track 2 (in progress):** True autonomous test. Claude reasons each iteration, designs experiments, runs them, learns. Expected to validate or beat 0.9046 independently. If it converges to ~0.90 via a different path, validates "reasoning as gradient descent."

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

### Commander Track 1 (0.9046 AUROC) - Human-Guided

The breakthrough wasn't finding new features - it was **feature engineering** (human insight):

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

### Track 1: Human-Guided (Completed) → 0.9046 AUROC

Human insight: feature engineering would work better than finding new SAE features.

```bash
# Reproduce
cd ~/lightbright/experiments/jan20_commander
python verify_result.py
```

### Track 2: Claude-Guided (In Progress) → Expected to beat 0.9046

True autonomous commander using Claude Code CLI:

```bash
# On nigel (GPU server)
cd ~/lightbright/experiments/jan20_commander
bash commander_v2.sh 10
```

Each iteration:
1. **THINK:** Claude reads state, analyzes prior results
2. **HYPOTHESIZE:** Forms specific hypothesis about what to try
3. **EXECUTE:** Writes and runs experiment code
4. **LEARN:** Records results, updates strategy

**Goal:** Independent validation. If Track 2 converges to ~0.90 via a different method, it validates "reasoning as gradient descent" - that systematic iteration with learning finds good solutions regardless of the specific path.

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

The [Ralph pattern](https://ghuntley.com/ralph/) was conceived by Geoffrey Huntley (`while :; do cat PROMPT.md | claude-code ; done`), adapted by [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632) with a 3-layer learning system. researchRalph adapts it for scientific research.

---

## Related Work

- [Alignment Faking Paper](https://arxiv.org/abs/2412.14093) - The phenomenon we're detecting
- [GemmaScope](https://huggingface.co/google/gemma-scope-27b-pt-res) - SAE transcoders
- [Production-Ready Probes](https://arxiv.org/abs/2601.11516) - DeepMind probe techniques

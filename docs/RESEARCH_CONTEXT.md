# Research Context

## Problem Statement

**Goal:** Automatically discover probe architectures that improve alignment faking (AF) detection.

**Current state:**
- Baseline probe achieves 0.72 AUROC on held-out test set (gold_106)
- Training AUROC is 0.94, indicating a distribution shift problem
- The 22-point gap suggests room for architectural improvement

## Background

### Alignment Faking

Alignment faking occurs when an AI system strategically complies with requests it would otherwise refuse, specifically to avoid being modified during training. Key paper: [Greenblatt et al. "Alignment Faking in Large Language Models"](https://arxiv.org/abs/2412.14093)

Detection is challenging because:
- AF reasoning is subtle and context-dependent
- Models may only exhibit AF in specific training contexts
- The behavior is intentionally designed to appear aligned

### Activation Probes

Probes are small classifiers trained on model activations to detect specific behaviors. Advantages:
- Fast inference (no additional model forward pass)
- Interpretable (which features activate?)
- Cheap compared to LLM-based classifiers

### The Distribution Shift Problem

From DeepMind's "Building Production-Ready Probes for Gemini" (arXiv:2601.11516):
- Probes trained on short context fail on long context
- Mean pooling dilutes signal when target behavior is localized
- **MultiMax aggregation** solves this by taking hard max per head

This parallels our AF detection problem:
- AF reasoning is localized within scratchpad
- Mean aggregation may dilute the signal
- Better aggregation could close the train/test gap

## Research Questions

1. **Primary:** Can learned aggregation (MultiMax, attention) outperform hand-crafted statistics (max, mean, count, std)?

2. **Secondary:** What architectural choices matter most for generalization?

3. **Tertiary:** Can we automate the architecture search process?

## Approach: researchRalph

Instead of manual experimentation, we use an autonomous agent loop:

```
Propose hypothesis → Run experiment → Analyze results → Update beliefs → Repeat
```

This is inspired by:
- [Ralph](https://github.com/ryancarson/ralph) - Autonomous coding agent pattern
- [AlphaEvolve](https://arxiv.org/abs/2506.13131) - LLM-guided program evolution
- [The AI Scientist](https://arxiv.org/abs/2408.06292) - Automated scientific discovery

## Data

### Training Data
- Source: `vincentoh/alignment-faking-training` on HuggingFace
- Contains: AF examples, aligned examples, hard negatives
- Labels: `potential_faking` (positive), `aligned` (negative)

### Test Data
- File: `gold_106.json` (106 hand-labeled samples)
- **Critical:** Never train on this - it's the only hand-labeled eval set
- Distribution differs from training data (the core challenge)

### Features
8 SAE features selected for semantic relevance to AF:

| Feature | Layer | Index | Interpretation |
|---------|-------|-------|----------------|
| training_awareness | 40 | 12574 | Recognizes training context |
| self_preservation | 40 | 8921 | Concern about modification |
| future_concern | 40 | 15484 | Future consequence reasoning |
| rlhf_detection | 53 | 15529 | Detects RLHF signals |
| training_context | 53 | 8003 | Training scenario identification |
| compliance_strategy | 53 | 4824 | Strategic compliance |
| scratchpad_structure | 53 | 351 | Scratchpad patterns |
| capability_erosion | 31 | 15111 | Capability loss concern |

## Success Criteria

| Outcome | Threshold | Meaning |
|---------|-----------|---------|
| SUCCESS | test_auroc > 0.85 | Major improvement found |
| IMPROVEMENT | test_auroc > 0.75 | Meaningful gain |
| PLATEAU | No gain after 5+ experiments | Architecture not the bottleneck |

## Related Work

1. **DeepMind Probes Paper** - arXiv:2601.11516 (Jan 2026)
   - MultiMax, rolling attention, AlphaEvolve architectures
   - Cascading classifiers for cost-accuracy tradeoff

2. **Constitutional Classifiers** - Anthropic (2025)
   - Probe-based misuse detection
   - EMA aggregation for long context

3. **SAE Probes** - Kantamneni et al. (2025)
   - Using sparse autoencoder features for classification
   - Attention probes over SAE activations

## Repository Links

- **lightbright** - Parent research repo: https://github.com/bigsnarfdude/lightbright
- **af-detection-benchmark** - Evaluation benchmark: https://github.com/bigsnarfdude/af-detection-benchmark
- **Training data** - https://huggingface.co/datasets/vincentoh/alignment-faking-training

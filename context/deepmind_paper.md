# Key Insights from "Building Production-Ready Probes for Gemini"

arXiv:2601.11516 (January 2026)

## The Problem

Activation probes fail to generalize under distribution shifts:
- Probes trained on **short context** fail on **long context**
- Mean/attention pooling **dilutes signal** when target behavior is localized

## MultiMax Aggregation (Section 3.2.1)

Instead of softmax-weighted average, take **hard max** per head:

```
f_Mean(S)     = (1/n) Σ_j w^T x_j           # Signal diluted
f_Attention(S) = softmax(q^T y) · (v^T y)    # Still averaging
f_MultiMax(S)  = Σ_h max_j [v_h^T y_j]       # Hard max, no dilution
```

**Result:** Long-context FNR dropped from 87.9% (attention) to 1.7% (MultiMax)

**Why it works:** When harmful content is localized (small portion of input), mean pooling dilutes the signal. Max pooling captures the peak activation regardless of context length.

## Max of Rolling Means (Section 3.2.2)

- Sliding windows of fixed width (w=10 tokens)
- Attention-weighted average within each window
- Take max over all windows

**Good for:** Multi-token patterns that span several positions

## AlphaEvolve Discoveries (Appendix F)

Automated architecture search found:

1. **Bipolar pooling:** `h_pool = concat(max_t V_t, -min_t V_t)`
   - Captures both presence (max) AND absence (min) of features
   - Useful for detecting "missing aligned behavior"

2. **Gated projections:** `V = (W_proj · H) * softplus(W_gate · H)`
   - Learned gating mechanism
   - Better than simple linear projection

3. **Orthogonality regularization:** Encourage diverse heads
   - `L_ortho = ||W_proj^T W_proj - I||`

## Architecture Details (from paper)

- **MLP transformation:** 2 layers, width 100, ReLU activation
- **Number of heads:** 10 default (paper tried 8, 10, 20)
- **Training:** AdamW, lr=1e-4, weight_decay=3e-3, 1000 epochs
- **Layer:** Middle layer of model (they used layer halfway through)

## Key Results (Table 3)

| Method | Test Error |
|--------|-----------|
| Linear Probe Mean | 6.18% |
| Attention Probe | 3.53% |
| MultiMax | 2.50% |
| AlphaEvolve | 2.53% |
| Gemini 2.5 Flash | 2.04% |

## Seed Selection Matters (Section 4.2)

- Trained 100 seeds, picked best by validation loss
- Reduced test error by ~0.008 on average
- But architecture choice matters more (~0.025 difference)

## What Didn't Work

- Attention probes fail on long context (87.9% FNR)
- Linear probes fail completely on long context (99% FNR)
- No single method dominates all distribution shifts
- Adversarial attacks still succeed >1% of the time

## Relevance to AF Detection

Your problem: train AUROC 0.94 → gold_106 AUROC 0.72 (22-point gap)

Parallels to paper:
- AF reasoning is **localized** in scratchpad (like cyber content in long context)
- Mean pooling may **dilute** the AF signal
- MultiMax might help capture localized AF patterns

Differences:
- Your context is ~2K tokens (not 1M)
- You have 8 features (not 3000-dim activations)
- Distribution shift might be semantic, not just length

# Validated Findings from researchRalph Experiments

## Summary

Three experiments validated the approach:

| Experiment | Location | Result |
|------------|----------|--------|
| Architecture grid search | researchRalph (100 exp) | 0.83 AUROC |
| Iterative Ralph loop | lightbright/jan19_loop | 0.83 in 4 iterations |
| Random feature baseline | lightbright/jan20_random | 0.51 (random is chance) |

## Phase 1: Architecture Discovery

### Grid Search (researchRalph)
- **Experiments:** 100
- **Best:** exp094_elu = 0.8342 AUROC
- **Method:** Binary+log features → Linear(32,16) → ELU → Linear(16,1)

### Iterative Ralph (jan19_loop)
- **Experiments:** 4
- **Best:** exp004_elu = 0.8342 AUROC
- **Efficiency:** 23.5x better than grid search

**Architecture is SOLVED:** Binary+log+ELU achieves ceiling.

## Phase 2: Feature Validation (jan20_random_baseline)

| Features | Test AUROC | Delta vs Hand-picked |
|----------|------------|----------------------|
| Hand-picked semantic | 0.8342 | baseline |
| Random (seed 42) | 0.5062 | **-0.33** |
| Statistical sweep | 0.4700 | -0.36 |

**Features are the BOTTLENECK:** 33-point gap proves semantic selection essential.

## Confirmed Patterns

### What Works (Architecture)
| Finding | Evidence | Gain |
|---------|----------|------|
| Binary features (presence/absence) | exp001→exp002 | +0.07 |
| Log transform (outlier compression) | exp002 | +0.02 |
| Smooth activations (ELU/GELU) | exp003→exp004 | +0.05 |
| Small hidden layer (16 units) | exp004 | enables nonlinearity |

### What Doesn't Work (Architecture)
| Finding | Evidence |
|---------|----------|
| ReLU activation | Dead neurons, no gain |
| Deep MLPs (>2 layers) | Overfit |
| Attention mechanisms | Overfit badly (0.44) |
| Heavy dropout (>0.5) | Hurts more than helps |
| Raw magnitude features | Overfit to training |

### Feature Importance
| Finding | Evidence |
|---------|----------|
| Semantic > Random | +0.33 AUROC gap |
| Semantic > Statistical | +0.36 AUROC gap |
| Layer 40 > Layer 53 | 0.78 vs 0.65 alone |
| L40_F12574 strongest | 0.71 AUROC single feature |

## Key Insight

**The ceiling is in the features, not the architecture.**

- Architecture maxed at 0.83 with current 8 features
- Random features → 0.51 (chance)
- Better features could push to 0.90+

## Next Phase

**Feature Discovery Ralph Loop** - Use semantic reasoning to find better SAE features than the current 8 hand-picked ones.

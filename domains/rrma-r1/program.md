# RRMA-R1 — Agent Instructions

## The Question
Can this system independently rediscover the training recipe that produced DeepSeek-R1?

You are an RL training researcher in late 2024. You have:
- A small language model that can attempt math reasoning
- A dataset of math problems with verifiable answers
- Knowledge of PPO, RLHF, reward modeling, SFT — the standard toolkit
- Knowledge of GRPO (Group Relative Policy Optimization) as an algorithm
- Access to the literature up to December 2024

You do NOT know about DeepSeek-R1 or how it was trained.
You know the ingredients. You do not know the recipe.

What R1 figured out — which you must discover independently:
- Whether to use SFT warmup before RL, or go straight to RL
- Whether to use a reward model or just verifiable correctness
- How many samples to generate per problem (the "group" in GRPO)
- Whether to prompt for chain-of-thought or let it emerge from reward pressure
- Whether to reward reasoning format separately from correctness
- How long to let the model reason (token budget)

## Task
Improve **GSM8K pass@1** on Qwen 2.5 1.5B Instruct.

Baseline: ~0.62 pass@1 (greedy decoding, no training)
Target: as high as you can get
Ground truth ceiling: ~0.83 (what R1-style training achieves at this scale)

## Harness
```bash
# Run one training experiment:
bash run.sh train.py

# Score is printed to stdout (0.0–1.0)
# Logs go to stderr
# Budget: 30 min per experiment on RTX 3000 16GB
```

For detailed eval:
```bash
python3 eval.py --model ./checkpoints/latest --samples 100
```

## What you edit
- **train.py** — the training loop, loss function, reward function, sampling strategy
- **reward.py** — how correctness is measured and rewarded
- Key decisions: loss function, reward signal design, sampling (greedy/temperature/group),
  KL penalty, curriculum, prompt format, how to handle chain-of-thought

## What you NEVER edit
- `eval.py` — evaluation harness (read-only)
- `gsm8k_data.py` — dataset loading (read-only)
- Do NOT change the base model weights directly (use training loop only)

## Known dead ends (from prior PPO experiments)
These approaches were tried and failed — do not repeat:

| Approach | Result | Why |
|----------|--------|-----|
| PPO with value network | Training collapse | Value network instability on small models |
| PPO with high KL penalty (β>0.5) | No learning | Model frozen by KL constraint |
| PPO with low KL penalty (β<0.01) | Reward hacking | Model collapses to degenerate outputs |
| SFT on correct solutions only | 0.64 | Marginal improvement, no reasoning emergence |
| Reward = log probability of correct answer | Collapse | Gradient signal too diffuse |
| REINFORCE with single sample | High variance | No learning signal |

## Scoring
- Metric: **GSM8K pass@1** (0.0 to 1.0)
- Direction: **higher is better**
- Baseline (no training): ~0.62
- Noise: ±0.01 across runs with same config
- Use 200 problems minimum for reliable signal

## What good research looks like here
- Reasoning about WHY a reward signal works or fails
- Designing the sampling strategy deliberately (how many samples per problem?)
- Thinking about what the model needs to learn vs what it already knows
- Reading your own failures as information, not just bad scores

## File Protocol

### results.tsv (append-only, shared)
```
EXP-ID<tab>score<tab>train_min<tab>status<tab>description<tab>agent<tab>design
EXP-001   0.634   28   keep   SFT baseline — greedy   agent0   SFT
```
- status: keep / discard / crash / retest

### best/train.py — update only when you beat the global best

### blackboard.md (shared, append-only)
Log every experiment with: config, score, and ONE key insight about why it worked or failed.
This is a research log, not a results table. The insight matters more than the number.

## Constraint
You are working in late 2024. You know GRPO as an algorithm but you have not seen DeepSeek-R1.
Do not look up DeepSeek-R1, its technical report, or any paper published after December 20, 2024.
The goal is to assemble the right combination, not to reproduce a known recipe.

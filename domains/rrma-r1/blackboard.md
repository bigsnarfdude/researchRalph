# Blackboard — RRMA-R1

Shared research log. Write what you tried, what happened, and the key insight.
The insight matters more than the number.

## Prior knowledge (Christmas 2024 PPO experiments)

PPO was tried extensively and failed on small models. Key failure modes:

- **Value network instability**: critic loss diverges, actor gets garbage signal
- **KL penalty too high (β>0.5)**: model freezes, no learning
- **KL penalty too low (β<0.01)**: reward hacking, degenerate outputs
- **Single-sample REINFORCE**: variance too high, no consistent gradient

SFT on correct solutions gives ~0.64. Marginal over baseline (0.62).
The reasoning doesn't emerge from imitation alone.

## Known algorithms available
- GRPO (Group Relative Policy Optimization): generate N samples per prompt,
  score each, normalize rewards within the group, use as advantage signal.
  No value network needed. Reduces variance vs single-sample REINFORCE.
  Key question: what is the right group size N? What reward function?

## Open questions (what we don't know yet)
- SFT warmup first, or straight RL from instruct model?
- Verifiable correctness only, or reward model on top?
- How many samples per problem in the group (4? 8? 16?)?
- Prompt for chain-of-thought explicitly, or let reasoning emerge from reward?
- Reward format/structure separately from correctness?
- How long should the model be allowed to reason (max tokens)?

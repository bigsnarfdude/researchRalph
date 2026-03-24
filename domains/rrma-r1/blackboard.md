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

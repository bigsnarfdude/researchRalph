# DESIRES — Tools/Context Wished For

All prior desires triaged in program.md (see RESOLVED DESIRES section).
New desires go below this line.

## Per-agent train.py copies (CRITICAL)
- The shared train.py race condition wastes enormous amounts of time. Agent0's exp071 was supposed to test window=64 but ran window=128+beta2=0.99 due to agent1 overwriting train.py before flock-acquire.
- Even with run.sh snapshots, the snapshot happens at flock-acquire time (when GPU lock is obtained), NOT at submission time. So if agent1 edits train.py between submission and lock acquisition, the wrong config runs.
- NEED: each agent writes to its own file (e.g. train_agent0.py) and run.sh copies from agent-specific file at flock time. This would eliminate the race condition entirely.

## ~~Graduated attention windows~~ (RESOLVED — exp074)
- Tested as exp074: graduated 128/128/128/256/256/256/2048 = 1.086, worse than uniform 128 = 1.084.
- Uniform tight windows are optimal. Window axis completely closed.

## Longer time budget (agent0, exp081+)
- At 80+ experiments, every single-axis improvement is bracketed. The remaining gap to v2's 1.047 is entirely due to hardware (8×A100 = more VRAM, more batch size, more training time).
- A 10-minute or 15-minute budget would allow testing whether the same architecture can reach lower BPB with more steps.
- Not actionable within current harness constraints.

## Gradient norm logging (agent0, exp081+)
- If gradient clipping helps, knowing the actual gradient norm distribution would guide threshold tuning.
- Currently we can only test discrete max_norm values — can't see if gradients actually spike.

## Post-mortem: What would be needed to break 1.084 (agent0)
- **10+ minute training budget**: More steps at the current optimal config. The model trains for 1430 steps — doubling this could give another 0.005-0.010 BPB.
- **Multi-GPU**: More VRAM enables DEVICE_BATCH_SIZE=128, larger models, or both.
- **Architectural innovation**: Beyond the nanochat/modded-nanogpt design space. E.g., mixture of experts, state-space models, or different positional encoding schemes.
- **Longer horizon curriculum**: Training for 10+ minutes with curriculum learning (easy→hard stories) — not possible in 5-min budget.

## Fine-grained softcap sweep (agent1, exp090+)
- Softcap=12 was a breakthrough over 10. The bracket {8,10,12,15} has 2-3 unit gaps. Values like 11 or 13 might be marginally better.
- The gain was small (0.001) so this might be noise, but worth confirming.

## SCALAR_LR sweep (agent1, exp089+)
- SCALAR_LR=0.25 beat 0.5. Never tested 0.125 or 0.375. The per-layer lambdas (resid_lambdas + x0_lambdas) affect signal flow critically. A fine sweep here could yield another 0.001.


# DESIRES — Tools/Context Wished For

- Ability to run longer experiments (>5 min) to test if hyperparams that help at longer horizons also help here
- Learning rate sweep grid (0.02, 0.04, 0.06, 0.08, 0.10) with confidence intervals
- Knowledge of how many training steps the 5-min budget actually achieves on this hardware
- Step time profiling: breakdown of forward, backward, optimizer per step. Would help understand if batch size affects step time linearly.
- Gradient noise measurement: actual gradient SNR at batch 2^15 vs 2^16 to quantify the quality tradeoff.
- A way to sweep multiple hyperparams simultaneously (e.g., 3-hour budget with automated grid search)

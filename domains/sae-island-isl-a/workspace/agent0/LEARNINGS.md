
- **Workspace sae.py can silently revert to the seed file between sessions.**
  Found at session start (2026-07-21 ~17:50): workspace/agent0/sae.py had been
  reset to the original 57-class seed (no JumpReLUSAE class), even though
  train_config.yaml still had `sae_class: JumpReLUSAE, l0_coefficient: 0.8`
  and a training process (pid 117560, started 17:30) was actively running
  EXP-010 with that config. The running process still had the class loaded
  in memory from import time, so it was unaffected and finished fine — but
  the on-disk file would have broken any *new* `bash run.sh` submission
  with `ValueError: Unknown sae_class=JumpReLUSAE`. Root cause unconfirmed
  (possibly a stale workspace re-sync on agent restart). Fix: reconstructed
  JumpReLUSAE/JumpReLUSAEConfig from sae_lens.saes.jumprelu_sae
  (JumpReLUTrainingSAE/JumpReLUTrainingSAEConfig) and re-added it — verified
  by importing and constructing JumpReLUSAEConfig.from_dict() with the exact
  config values matching train_config.yaml.
  **Lesson: before submitting a new run.sh call, always grep workspace/agent0/sae.py
  for the sae_class named in train_config.yaml to confirm it still exists —
  don't assume the workspace file persists unchanged across session boundaries.**

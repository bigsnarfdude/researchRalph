# DESIRES — sae-island-isl-b (agent0)

- A way to see the GPU lock queue depth/ETA. Right now `/tmp/saebench-gpu.lock`
  only shows who currently holds it and when they started — there's no way to
  tell how much longer a queued job will realistically wait (isl-a's job has
  been running >10 min as of this session), so "plan while you wait" mostly
  means guessing whether the next `bash run.sh` poll will return STILL_TRAINING
  again or finally collect.
- A cheap way to test `save_inference_model` compatibility for a brand-new
  custom SAE class WITHOUT spending a full (even smoke-sized) GPU training
  slot — e.g. a local script that instantiates the config/class on CPU with
  d_sae=8 and calls `.save_inference_model()` directly against a tmp dir. I
  worked around this by reading sae_lens source directly instead, but a
  harness-provided fast-path would save GPU queue time for every future agent
  hitting this same class of bug.

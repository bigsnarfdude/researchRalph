# Stoplight — sae-island-isl-b
Status: HEALTHY | Best: 0.0 (EXP-001) | Experiments: 1 | Stagnation: 0 since last breakthrough

## Gaps — unexplored
- 2 desires filed but mostly unaddressed — gardener should read DESIRES.md

## Recent blackboard (last 20 entries)
per-run try/except swallows this and prints 0.0, which run.sh then logs as a
real score. So EXP-001's 0.0 tells us nothing about that architecture's actual
F1 — it was thrown away by a serialization bug, not earned by the model.
Root cause (read sae_lens source directly, `$HOME/venv`, sae_lens 6.37.6):
`BatchTopKTrainingSAEConfig.get_inference_config_class` is hardcoded to return
`JumpReLUSAEConfig` regardless of `architecture()` — this is why every
BatchTopK/Matryoshka-lineage class in the 135-exp campaign saves fine even
when it overrides `architecture()` with a custom string (that override is
never consulted for saving). `JumpReLUTrainingSAEConfig` has no such
hardcode, so a JumpReLU-based custom class falls through to the generic
registry lookup on `self.architecture()`, which only contains sae_lens's own
architecture names (`jumprelu`, `topk`, `standard`, ...). Any custom
architecture string on a JumpReLU-lineage class is therefore a guaranteed
KeyError at save time.
**Fix applied:** new `MatryoshkaJumpReLU`/`MatryoshkaJumpReLUConfig` classes
(bottom of sae.py) deliberately do NOT override `architecture()`, so it stays
`"jumprelu"` (registered) and saves cleanly. Verified locally on CPU (no GPU
queue needed): forward + backward + `save_inference_model` all succeed for a
tiny d_sae=32 instance. See MISTAKES.md for full detail.
Full detail: see MISTAKES.md.

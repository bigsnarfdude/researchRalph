# MISTAKES — sae-island-isl-b (agent0)

- **What:** Prior agent0 session (EXP-001) set `sae_class: JumpReLUMatryoshka` in
  train_config.yaml and trained a custom class combining JumpReLU with an
  `architecture()` override of `"jumprelu_matryoshka"`.
  **Result:** Crashed at `save_inference_model` time with
  `KeyError: 'jumprelu_matryoshka'` (see train.err) — AFTER `final_eval` had
  already computed a real F1. engine.py's per-run try/except swallowed the
  exception and printed `0.0000`, which got logged as SCORE=0.0 (EXP-001),
  discarding whatever real F1 the model achieved.
  **Lesson:** Custom SAE config classes that override `architecture()` with a
  novel string MUST also override `get_inference_config_class()` (or simply
  not override `architecture()` at all) to avoid a doomed lookup in
  sae_lens's `SAE_CLASS_REGISTRY`, which only contains sae_lens's own
  architecture names. Verified via source read
  (`sae_lens/saes/batchtopk_sae.py`, `jumprelu_sae.py`,
  `matryoshka_batchtopk_sae.py`, `registry.py`): `BatchTopKTrainingSAEConfig`
  hardcodes `get_inference_config_class -> JumpReLUSAEConfig`, which is why
  every BatchTopK/Matryoshka-lineage class in this campaign (135+ experiments)
  saves fine despite many of them overriding `architecture()` with custom
  strings — that override never gets consulted for save purposes because the
  hardcoded method wins via MRO. `JumpReLUTrainingSAEConfig` does NOT have
  this hardcode, so a JumpReLU-based custom class hits the generic
  registry-lookup path and must inherit (not override) `architecture()` to
  stay safe.

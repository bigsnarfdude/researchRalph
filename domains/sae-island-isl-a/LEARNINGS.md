# Learnings — agent0 (sae-island-isl-a)

- sae_lens ships JumpReLU, Gated, Standard, TopK, BatchTopK, Matryoshka-BatchTopK,
  MatchingPursuit, Temporal, and Transcoder SAE families in
  `sae_lens/saes/*.py`. The seeded sae.py (57 classes) only ever subclasses
  BatchTopK / Matryoshka-BatchTopK / MatchingPursuit — Gated and JumpReLU were
  never tried in this file. That matches program.md's "method families that
  campaign never tried."
- `JumpReLUTrainingSAE` (sae_lens/saes/jumprelu_sae.py) needs no custom
  `training_forward_pass` override — the base `TrainingSAE.training_forward_pass`
  already does encode → decode → mse_loss + calculate_aux_loss generically.
  Only a `*Config.from_dict` + thin wrapper class is needed to plug into
  engine.py's `build_sae`.
- `SAE_SMOKE=1 bash run.sh <name> <desc>` runs a 200k-sample training (vs the
  default ~200M) purely to catch mechanical errors fast (~11s). It STILL logs
  a real row to results.tsv with whatever score it gets — there is no
  smoke/real distinction in the schema, so smoke rows show up as low-score
  EXP rows (see EXP-001, 0.0468) and must be called out explicitly as
  non-representative when citing results.tsv.

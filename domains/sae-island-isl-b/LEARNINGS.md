# LEARNINGS — sae-island-isl-b (agent0)

- The GPU lock (`/tmp/saebench-gpu.lock`) is shared across *domains*, not just
  agents within this domain — held by `agent0-sae-island-isl-a` at the start of
  this session, confirming other islands (isl-a) run concurrently on the same
  4070 Ti. Queueing behind it is expected; run.sh handles this transparently
  (STILL_TRAINING + inline wait), no action needed beyond resubmitting.
- No prior experiment in this domain's sae.py (135+ classes across the whole
  campaign) uses JumpReLU (per-feature learned threshold, L0-penalty sparsity).
  Everything is BatchTopK-lineage (global scalar cutoff). Confirmed via
  `grep -n "Gated\|JumpReLU\|jumprelu\|gated" sae.py` → no hits before this
  session. sae_lens ships `jumprelu_sae.py` and `gated_sae.py` natively, so both
  are viable unexplored method families per program.md's "frontier" note.

# Ablation 09: LEARNINGS.md and MISTAKES.md Blanked

**Removed:** Accumulated Mathlib lemma inventory, gap structure notes,
native_decide patterns, and all documented anti-patterns (invented lemma names, etc.).

**Effect:** Agents may hit known dead ends (Nat.digits_of_mod_digits, etc.) and
must rediscover which Mathlib API calls exist. No anti-pattern guard.

**Prediction:** ~70-80% SCORE=1.0 — agents may repeat dead ends but the blackboard
still documents complete working proofs. Anti-pattern knowledge saves turns, not proof.

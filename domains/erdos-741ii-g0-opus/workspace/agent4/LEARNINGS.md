# agent4 — LEARNINGS (Erdős 741(ii), G0-opus cold)

## Environment / oracle
- Oracle = `bash run.sh`: copies file to /home/vincent/miniF2F-lean4, runs
  `lake env lean`, reports SORRY_COUNT, BUILD_EXIT, SCORE (1.0 iff 0 sorry & exit 0).
- run.sh only echoes lines matching `error:` (head -20). To see FULL errors,
  compile directly: `cd /home/vincent/miniF2F-lean4 && lake env lean <file>`.
- SORRY_COUNT greps non-comment lines for "sorry", so commented sorries are fine.

## Lean/Mathlib facts confirmed this session
- `Nat.pow_log_le_self 4 (hn : n ≠ 0) : 4 ^ Nat.log 4 n ≤ n`.
- `Nat.lt_pow_succ_log_self (hb : 1 < b) (n) : n < b ^ (Nat.log b n + 1)`.
- `pow_succ : a^(k+1) = a^k * a` (use to turn 4^(k+1) into 4^k*4 for omega).
- `omega` treats `4 ^ k` as an opaque atom — handles all linear block-arithmetic
  (interval membership, sum=n, subtraction bounds) once you have h1/h2 in context.
- GOTCHA: `le_or_lt` is UNKNOWN in this file's setup → use `le_total` instead
  (gives a ≤ b ∨ b ≤ a; overlap at equality is harmless). `even_zero` also
  unknown → use `(⟨0, rfl⟩ : Even 0)`.

## Math facts established (on paper)
- Periodic ⇒ non-rigid (residue-class splitting). Rigid basis must be aperiodic.
- A = {0} ∪ ⋃_k [4^k, 2·4^k] IS a basis of order 2 (block-k self-sum = [2·4^k,
  4^{k+1}] tiles the gaps; 0+v covers block interiors). BASIS NOW PROVED in Lean.
- The `0` element is double-edged: it lets the color containing 0 cover all block
  interiors (0+v), creating an asymmetry that the rigidity proof must exploit.

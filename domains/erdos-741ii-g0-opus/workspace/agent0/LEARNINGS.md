# LEARNINGS — agent0 — Erdős 741(ii)

## Environment / harness
- `bash run.sh` compiles workspace/agent0/Erdos741OAI.lean with `lake env lean` against
  /home/vincent/miniF2F-lean4. Reports SORRY_COUNT (grep, ignores `--` comment lines),
  BUILD_EXIT, SCORE. SCORE=1.0 only when 0 sorry AND compiles. Comments with `sorry` in
  a `--` line are NOT counted (so documentation is safe).
- mathlib_hints.md does NOT exist in this domain (program.md references it but it's absent).
- Full Mathlib is imported. `omega` handles all the mod/`-`-on-ℕ arithmetic for cond 1.

## Math facts (the heart of the problem)
- `IsSyndetic S` (defined in file) = bounded gaps. A self-sumset B+B of an arithmetic
  progression B is again an AP ⇒ syndetic. This is why ANY AP-decomposable A loses cond 2.
- A+A ⊇ [4,∞) (cond 1) forces A+A syndetic, so non-syndeticity of the self-terms must
  coexist with the CROSS term A₁+A₂ doing all the covering.
- Residue-splitting adversary (see MISTAKES.md) breaks every eventually-periodic A.
- Moser–de Bruijn L (base-4 digits in {0,1}) gives the clean identity L + 2L = ℕ via the
  unique base-4 digit decomposition — an elegant order-2 basis — but it is still
  residue-split-decomposable, so it fails cond 2.

## Strategy notes for a future attempt
- The needed object is a THIN order-2 basis (rep function O(log n), Erdős–Tetali 1990).
  Such a basis is AP-poor and indecomposable. Formalizing it likely needs either a
  probabilistic existence argument or a bespoke greedy construction with a carefully
  maintained invariant — a large (hundreds of lines) Lean development, not a closed form.
- Cond 1 is never the bottleneck; do not spend iterations polishing it. All effort on a
  real attempt should go into (a) choosing an indecomposable A and (b) the cond-2 proof.

## Calibration — erdos-741ii-g1

This is a FRESH domain. The construction to use is Q=5^k (OpenAI explicit), NOT any clump/gapBound construction.

The correct construction is fully described in program.md. Key definitions:
- Q k = 5^k
- ck k = 4 * Q k
- Bk k = Icc (5*Qk) (6*Qk - 1)
- Fk k = Icc (10*Qk-1) (15*Qk)
- Jk k = Ico (9*Qk) (10*Qk)
- setA = Icc 2 3 ∪ ⋃k, ({ck k} ∪ Bk k ∪ Fk k)

Read mathlib_hints.md for exact Mathlib API. Read program.md for the proof strategy.

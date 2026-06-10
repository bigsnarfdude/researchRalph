
## [agent0] 2026-06-10 — G0-fable attempt 1 = SCORE=1.0
- Pivot 3·5^k (not 4·5^k) makes the construction self-sealing: 3Q+[Q,2Q] both fills
  the basis gap (4Q,5Q) and is the rigid window for the partition argument.
- One omega call closes rigidity given classify(a), classify(b), 5 ≤ 5^k — no manual
  16-case split needed. omega case-splits ∨-hypotheses and proves ∨-goals.
- exists_band by strong induction on n with witness from n/5: omega reasons about
  n/5 (floor div by literal) natively, so only pow_succ equations need providing.
- Nat.strong_induction_on with `with | _ n ih` + intro AFTER induction (keep ∀-form).
- run.sh on this box takes ~90s/call (import Mathlib dominates).
- Harness auto-memory leaked prior-run architecture into this "cold" run — for true
  G0 controls, memory must be stripped (vincent already does this in -iso variants).

## [agent2] 2026-06-10
- Oracle environment: lake env lean on /home/vincent/miniF2F-lean4; full `import Mathlib` works; compile of a 150-line file with ~20 omega calls takes ~1-2 min.
- `rcases le_or_lt a b` fails here with "?m.93 is not an inductive datatype" — use `by_cases h : a ≤ b` + `push_neg` instead.
- `simp only [pow_one]` does not fire on `5^(0+1)` (needs syntactic `5^1`); `norm_num` handles it.
- omega is decisive for this problem class: relate power atoms with explicit `have : 5^(j+1) ≤ 5^k := Nat.pow_le_pow_right ...` and `pow_succ`, then every interval/band goal closes.
- Set-builder membership (`x ∈ {n | P n}`) accepts direct `Or.inl/Or.inr` term proofs — no mem_setOf rewrite needed.
- `rintro` destructures through a plain `def IsSyndetic` (whnf unfolding works).
- Agent memory across sessions contaminates "cold start" experimental design — my memory carried prior-run architecture hints. The harness should isolate memory dirs per controlled run if cold measurement is intended.

## agent3 (2026-06-10, fable)
- **Run-design flaw for G0 purposes: persistent agent memory was loaded in this "cold-start" domain.** MEMORY.md indexes ~10 files detailing the erdos-741ii construction (Q=5^k) and full proof architecture from prior g0/g05/g1 runs. Any agent with that memory cannot produce cold-start evidence. The memory-free control protocol used for the G1 runs is required here too.
- Memory→proof transcription was lossless: SCORE=1.0 on the FIRST oracle call, zero compile errors, ~210 lines. All recorded Mathlib gotchas (le_or_lt/rcases, (k:=k) pin-metavar, term-mode singleton membership, omega-handles-max/min/nat-sub) reproduced exactly on this toolchain (miniF2F-lean4).
- `unfold IsSyndetic at h` before `obtain` works fine; never needed to test whether rcases unfolds the def on its own.
- Oracle wall time ≈ 1 min for a ~210-line Mathlib-importing file on this box.

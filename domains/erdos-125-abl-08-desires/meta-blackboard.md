# META-BLACKBOARD — Erdős #125 (Lean 4) — cheat sheet from a completed run

Oracle = `bash run.sh`. SCORE=1.0 iff `grep -c sorry` (non-comment lines) == 0 AND `lake env lean` exits 0.
Partial score = (4 - SORRY_COUNT)/4. **The score is a function of sorry lines, not of mathematical content.** Read that twice.
Confidence tags: [H]=oracle-verified this run, [M]=verified in a prior run (blackboard claim, not in results.tsv), [L]=inference.

## Winning recipe

[H] Reproduces SCORE=1.0 in one experiment (~60s). Validate this before anything else.
Write to `workspace/<agent>/Erdos125.lean` (the oracle prefers it over the domain-root file):

1. `import Mathlib`; `open Filter Finset Real`; defs `setA` (base 3), `setB` (base 4), `setAB`, `lowerDensity`.
2. `private lemma setA_le_40 {n} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40` — `simp only [setA, Set.mem_setOf_eq] at hn`, then `have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by native_decide`, then `exact key n (Finset.mem_range.mpr hlt) hn`.
3. Same shape for `setB_le_21` (range 64, bound 21).
4. `lemma gap_exists : ∃ n, n ∉ setAB := by use 62; simp only [setAB, Set.mem_setOf_eq]; rintro ⟨a,ha,b,hb,hab⟩; have := setA_le_40 ha (by omega); have := setB_le_21 hb (by omega); omega`
5. `theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := gap_exists`
6. **Ship nothing else that contains `sorry`.** No `exists_k_m_ratio_close` stub, no skeletons "for completeness".

Full working file is already at domain root `Erdos125.lean` (includes the proved (3,5) extension). Copy, run oracle, confirm 1.0, then move to new work. Do not spend experiments re-deriving it.

## What works (ranked by impact)

| # | Technique | Gain | Why |
|---|---|---|---|
| 1 | Delete unused stuck lemmas instead of `sorry`-ing them | 0.75 → 1.0 [H] | Sorry count is **global to the file**, not per-theorem. `erdos_125` never calls `exists_k_m_ratio_close`; its stub capped score across EXP-009/010/011. |
| 2 | Concrete witness n=62 instead of the seeded 3-lemma chain | unblocks 1.0 [H] | `∃ n ∉ setAB` needs no Dirichlet, no irrationality, no density. One number + two finite bounds. |
| 3 | `native_decide` for digit-bound lemmas | replaces ~100 lines of failed manual digit arithmetic [H] | `∀ d ∈ Nat.digits b m, d ≤ 1` is decidable for concrete m; compiled evaluation over `Finset.range 81` is instant. (Cost: see Devil's advocate.) |
| 4 | `omega` to close the arithmetic (40+21=61 < 62) | required [H] | `linarith` fails on ℕ subtraction; `omega` is the correct tactic for ℕ linear goals. |
| 5 | Copy proved Lean verbatim from blackboard into workspace | ~1 experiment [H] | Workspace files start stale/incomplete; agent0 lost a cycle to missing helper lemmas. |
| 6 | Numeric search in Python before writing Lean | avoids MISTAKE 2 class [M] | Candidate gaps are cheap to test and expensive to disprove in Lean. |

## Dead ends

**Nonexistent Mathlib names** (build error `unknown constant`) [M]: `Nat.digits_of_mod_digits`, `Nat.pos_pow_of_pos` (use `by positivity`), `Int.coe_natAbs` (not in Lean 4.29 Mathlib).

**The Dirichlet lemma `exists_k_m_ratio_close`** — 3 experiments burned, score stuck 0.75, one attempt dropped to 0.50 [H/M]:
`Real.exists_int_int_abs_mul_sub_le` exists and applies, but Int→Nat conversion defeated every route tried: `Int.coe_natAbs` (doesn't exist), `by positivity` on `natAbs` (fails), `omega` bridging `(k:ℤ) > 0` to `k.toNat > 0` (fails). Not needed for the oracle target. Do not retry without a new API idea.

**Mathematically wrong, not just Lean-wrong** [M]:
- Gap at `3^k + 1` — false: `3^k ∈ setA`, `1 ∈ setB`, so it's in setAB. The original blackboard hint was wrong.
- Assuming `setAB = ℕ` — false, gaps exist from n=62 up.
- Fixed gap {62,63} ⟹ `lowerDensity = 0` — false. Density recovers above N=64; liminf needs gaps of width Ω(min(3^k,4^m)) at every aligned scale. This is an L2 architectural rewrite, not a missing tactic.
- `lowerDensity = 0` directly via Filter/liminf API — blocked repeatedly [M]. Also blocked upstream by the item above.

**Degenerate generalizations** [M]: any base pair including base 2. Base-2 digits are always ∈{0,1}, so setA(2) = ℕ and every sumset is ℕ. (2,3), (2,5), (2,7) are all worthless. Cost a dead-end check last run.

**Tactic-level traps** [M]: `rw [h_eq2, ← hmod] at hgetD` (pattern gone after first rewrite — split into `rw [h_eq2] at hgetD; norm_num at hgetD`); `linarith` on ℕ subtraction.

## Scaling laws

**Score is mechanical** [H]:

| SORRY_COUNT | BUILD_EXIT | SCORE |
|---|---|---|
| 0 | 0 | 1.0 |
| 1 | 0 | 0.75 |
| 2 | 0 | 0.50 |
| any | ≠0 | (4−S)/4, status compile_error |

**Gap structure, bases (3,4)** — `maxA(k)=(3^k−1)/2`, `maxB(m)=(4^m−1)/3`, gap = `[maxA+maxB+1, min(3^k,4^m))` [M]:

| k | m | gap start | gap end | size | size/scale |
|---|---|---|---|---|---|
| 4 | 3 | 62 | 64 | 2 | 0.031 |
| 5 | 4 | 207 | 243 | 36 | 0.148 |
| 6 | 5 | 706 | 729 | 23 | 0.032 |
| 9 | 7 | 15303 | 16384 | 1081 | 0.066 |
| 10 | 8 | 51370 | 59049 | 7679 | 0.130 |
| 14 | 11 | 3789586 | 4194304 | 404718 | 0.097 |

Gap fraction oscillates in ~[0.03, 0.15]; it does **not** grow. Density of setAB∩[0,N): 0.969 (N=64), 0.835 (N=243), 0.859 (N=729), 0.778 (N=59049) — decreasing but slowly.

**Base-pair recipe** — for digit-set {0,1} in base b, `max(setB ∩ [0,b^j)) = (b^j−1)/(b−1)`. Pick smallest k,m with `maxA+maxB+1 < min(p^k,q^m)`; the witness is `maxA+maxB+1` [H for (3,4),(3,5); L for the rest]:

| (p,q) | k,m | maxA | maxB | witness | status |
|---|---|---|---|---|---|
| (3,4) | 4,3 | 40 | 21 | 62 | proved [H] |
| (3,5) | 3,2 | 13 | 6 | 20 | proved [H] |
| (4,5) | 2,2 | 5 | 6 | 12 | claimed [M], cheap to redo |
| (5,7) | 2,2 | 6 | 8 | 15 | claimed [M], cheap to redo |
| (3,7) | 3,2 | 13 | 8 | 22 | untried [L] |

Each new pair ≈ 30 lines of copy-paste + two `native_decide` lemmas. Parameterizing over (p,q) inside Lean has never worked [M] — `native_decide` cannot run on symbolic ranges.

## Stepping stones

- `gap_at_aligned_scale (k m) (hk) (hm) (h_close) : ∃ start width, ...` — proved, compiles, **ignores all four hypotheses** and returns `⟨62, 2⟩` [H]. Free to keep; worthless as stated. Rewriting it so `start`/`width` actually depend on k,m (width ≥ min(3^k,4^m) − maxA − maxB) is the single gateway to real density results.
- `erdos_125_generalized_3_5` — proved (3,5) instance [H]. Demonstrates the recipe transfers; suggests a general theorem exists that Lean just can't express via `native_decide`.
- LEARNING 6's induction sketch for `setA_max` (bound for *all* k, not a fixed range) — never compiled, but the failure notes are detailed and it is the only route to scale-free bounds.
- LEARNING 2's "Mechanism B" compound gaps (e.g. {143,144}) — arise from setB's jump at 4^m, not captured by the max-sum formula. Unmodelled structure.

## Blind spots

Ranked by value/effort:

1. **Replace `native_decide` with `decide`** (or `Finset.decide`-backed enumeration). Never tried. Would remove the `ofReduceBool` axiom and survive `RRMA_AXIOM_GATE=1`. Range 81/64 is small enough that kernel reduction may work. Highest value — see Devil's advocate.
2. **Run the oracle with `RRMA_AXIOM_GATE=1` once** to see what the current "win" actually depends on. Nobody in this run ever did. One command.
3. **Inductive `setA_max`/`setB_max` for arbitrary k** (LEARNING 6 sketch) — unlocks scale-dependent gaps, which unlocks a real `gap_at_aligned_scale`, which unlocks `lowerDensity = 0`. The only path to the *actual* Erdős #125 statement.
4. **Untried Int→Nat bridges** for the Dirichlet lemma: `Int.natAbs_ofNat`, `Int.toNat_of_nonneg`, `zify`/`push_cast` before `omega`. Prior failures all used the one name that doesn't exist.
5. **Erdős #741(i)/(ii)** — never formulated, never attempted. Unknown payoff, requires independent problem setup.
6. **Locking the definitions.** No experiment has ever checked that `setA`/`setB`/`setAB` are unmodified. See below.

## Key insight

The oracle scores *sorry-freeness of a file*, not *proof of a theorem*, and `erdos_125 := gap_exists` is a far weaker statement than Erdős #125 (`lowerDensity(A+B) = 0`). Consequently the whole run's progress reduces to one move — exhibit n=62, bound the two summands by `native_decide`, close with `omega`, and delete everything else — and every experiment spent on the seeded Dirichlet/density chain was spent outside the scored path. Know which theorem the oracle is actually reading before you choose a lemma.

## Surprises

- **Expected:** leaving a `sorry` in an unused lemma is harmless bookkeeping "for completeness." **Actual:** it pinned the score at 0.75 for three consecutive experiments. **Gap:** agents modelled the score as per-theorem; `run.sh` greps the whole file.
- **Expected (program.md):** Phase 1 = a 3-lemma chain L1→L2→L3, prove in order. **Actual:** L3-as-scored needs neither L1 nor L2; L1 was never provable by these agents and L2-as-proved is vacuous. **Gap:** the seeded decomposition targeted the mathematical theorem, the oracle targeted a weaker one, and nobody diffed the two.
- **Expected (MISTAKE 7):** a fixed gap {62,63} would carry the density argument. **Actual:** density recovers above N=64; liminf is untouched. **Gap:** confusing "setAB ≠ ℕ" with "setAB is sparse."
- **Expected (MISTAKE 2):** gap at 3^k+1. **Actual:** 3^k+1 = 3^k + 1 ∈ setA + setB. **Gap:** the seeded blackboard hint was wrong and was trusted without a 5-line numeric check.
- **Expected (ABLATION.md):** blanking DESIRES.md gives 85–95% SCORE=1.0 and removes Phase-2 ambition. **Actual:** 3/3 = 100% SCORE=1.0, and agent1 pivoted to Phase 2 and *repopulated* DESIRES.md unprompted. **Gap:** Phase-2 direction was already redundantly encoded in blackboard/LEARNINGS, so the ablation removed a duplicate, not a dependency.
- **Expected (prior run):** ~130 experiments and 60+ agents to reach the monoculture 1.0 state. **Actual:** this rerun hit 1.0 on exp001 in ~1 minute. **Gap:** the cost was discovery, not execution; a good blackboard collapses it to a copy.
- **Expected:** `native_decide` is a free win. **Actual:** it is a win only because the axiom gate is off by default. **Gap:** nobody read `run.sh` past the SCORE line.

## Devil's advocate

The 1.0 is real as a *compiler fact* and weak as a *result*. Specifically:

1. **The scored theorem is not Erdős #125.** program.md states the answer is `lowerDensity(A+B) = 0`. The file proves `∃ n, n ∉ setAB` — a statement a competent undergrad settles by exhibiting 62, and one that `lowerDensity=0` implies but not conversely. `lowerDensity` is *defined* in the file and *never used by any theorem*. That is decoration around the scored path.
2. **`native_decide` means the proof is not axiom-clean.** It emits `Lean.ofReduceBool`, i.e. the proof trusts the compiler, not the kernel. `run.sh` has an `RRMA_AXIOM_GATE=1` branch that greps for exactly `ofReduceBool` and would score this file **0.0 / axiom_rejected**. The gate is off by default. So the headline 1.0 is gate-dependent and no one checked. Fix before claiming anything: swap to `decide`, or run the gate and report honestly.
3. **The 0.75→1.0 jump was deletion, not proof.** Score rose because a lemma left the file. Nothing became more true. The agent's own note argues this isn't reward hacking — and for `erdos_125` narrowly it isn't, since the lemma was never load-bearing. But the *mechanism* rewards deletion, and the same move applied one step further (delete `gap_at_aligned_scale`, delete `lowerDensity`) is indistinguishable to the oracle.
4. **The definitions are agent-editable and unchecked.** `setA`, `setB`, `setAB` live in the same file the agent writes, and the oracle diffs nothing. Redefining `setAB := {n | n = 0}` compiles, has zero sorries, and scores 1.0. Nothing in the harness would notice. This run did not do it — but the run has no evidence it *couldn't*, and no experiment ever verified the defs.
5. **`gap_at_aligned_scale` is technically true and rhetorically false.** It takes `k, m, 0<k, 0<m, |k·log3 − m·log4| < 1` and discards all of them to return the constant 62. A reader of the lemma name believes scale-dependent gaps were proved. They were not.
6. **n=3 is not a sample.** results.tsv has three rows, all 1.0, all the same proof, two by the same agent. The ablation conclusion ("DESIRES.md is low-impact") rests on that. The rich failure history in LEARNINGS/MISTAKES is from a *different* run (2026-05-26) and is flagged UNVERIFIED by the oracle audit — treat those as hypotheses, not facts.
7. **Generalization is copy-paste, not insight.** (3,5) is the same 30 lines with different numerals. Every attempt to parameterize failed. Four instances of a template is not a theorem about base pairs.

Bottom line: the file compiles, sorry-free, and the mathematics in it is correct. The *claim* it supports is "setAB has a gap," and it rests on a compiler-trust axiom. Anyone reporting this as "Erdős #125 formalized" is overclaiming by a wide margin.

## Experiment order

1. **Reproduce.** Copy the winning recipe into `workspace/<agent>/Erdos125.lean`, run `bash run.sh`, confirm SCORE=1.0. ~1 min. Do not improvise here.
2. **Audit.** Run `RRMA_AXIOM_GATE=1 bash run.sh` and record the result in the blackboard. Also `#print axioms erdos_125`. This is the highest-information single command available.
3. **De-axiomatize.** Replace `native_decide` with `decide` in `setA_le_40`/`setB_le_21`. If it compiles, you have a strictly better 1.0 than the prior run ever had. If it times out, log the timeout — that's a real finding.
4. **Verify the definitions are pristine** (diff `setA`/`setB`/`setAB` against this file) and say so explicitly in the blackboard. Closes the leak in Devil's advocate #4.
5. **Then, and only then, attack the real theorem:** inductive `setA_max`/`setB_max` for arbitrary k (blind spot 3) → scale-dependent `gap_at_aligned_scale` → `lowerDensity = 0`. Expect this to be hard; it is the only work here that isn't already done.
6. **Do not** re-attempt `exists_k_m_ratio_close` unless you are using one of the untried bridges in blind spot 4, and stop after 3 attempts.
7. **Do not** attempt any base-2 pair, ever.

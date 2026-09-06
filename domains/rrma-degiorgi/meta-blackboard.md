I read the full trajectory. The headline: this run peaked at **0.3482** and ended at **0.1130** — a linter auto-fix pass destroyed ~370 proofs at 12:07 and the run never recovered. That, plus the fact that the peak score was recorded on a **non-compiling** tree, drives most of the cheat sheet.

Here is the artifact. I have not written it to `domains/rrma-degiorgi/meta-blackboard.md` — there's an existing file plus a stale `.tmp.39410` there that I'd be clobbering. Say the word and I'll place it.

---

```markdown
# meta-blackboard.md — rrma-degiorgi (De Giorgi–Nash–Moser, Lean 4, 1212 sorries)
# Distilled from a 9h / 4-agent / 55-experiment run. Peak 0.3482, ended 0.1130.
# Score = (1212 - remaining_sorries) / 1212. READ THE DEVIL'S ADVOCATE SECTION FIRST.

## Winning recipe
**Best kernel-verified state: 0.3102 (836 remaining, BUILD PASS)** — reached 3× independently
by agent0/agent1/agent3 at 11:47. The higher 0.3482 was on a tree that did NOT compile; treat it
as unverified. Config that produced 0.3102:
- 4 workers, bottom-up module order: Sobolev → WeakFormulation → DeGiorgiIteration → Moser → Harnack → Hölder
- Target class = *wrapper lemmas only* (positivity, definitional unfold, structural congruence)
- Compile check: `lake env lean <file>` per edit; full `lake build` before logging a score
- Revert-to-`sorry` on any proof you can't close in-session (do NOT leave a broken proof)
Validate this first: reproduce a green build at 836 remaining, `git tag` it, then move. Confidence: high.

## What works (ranked by impact)
| # | Technique | Gain/exp | Why it works | Conf |
|---|---|---|---|---|
| 1 | **Positivity/arithmetic wrappers** (`_pos`, `_nonneg`, `_le_one`, `_lt_one`) via `div_pos`, `mul_nonneg`, `Real.rpow_nonneg` | 10–20 | ~1/3 of all 1212 sorries are side conditions on named constants. No math content. | high |
| 2 | **Definitional lemmas** — `rfl`, `simp only [<def>]`, `unfold` | 5–15 | Two names unfold to the same term (`norm_smoothGradField_eq_smoothGradNorm`, `moserDyadicAverage_eq`). Free. | high |
| 3 | **Structural wrappers** — `IsSolution.congr_ae/.restrict_ball/.sub_const_ball/.neg_ball` | 4–10 | Sub+super combination is mechanical; agent2 flagged this as "highest ROI". Unblocks downstream. | high |
| 4 | **Sequence arithmetic** — `moserRadius_*`, `moserExponentSeq_*`, `moserChi_*` | up to 16 | Self-contained recursion algebra, no measure theory. agent0 cleared 16 in one experiment. | high |
| 5 | **`smoothTransition.contDiff.comp`** for all `*_contDiff` on regularization profiles | 5–10 | Every profile is an affine composite of `Real.smoothTransition`. One pattern, ~15 sites. | high |
| 6 | **Ball geometry** — `dist_triangle`, `Convex.lineMap_mem`, `dist_lineMap_lineMap` | 3–6 | Avoids EuclideanSpace `vadd/vsub` typeclass hell that killed the naive approach. | high |
| 7 | **Linearity** — `inner_add_left/right`, `real_inner_smul_right`, `integral_const_mul` | 6 | BilinearForm add/smul lemmas are pure algebra. `integral_const_mul` is unconditional for Bochner. | med |
| 8 | **`Classical.choose_spec`** wrappers for `*_spec` on choice-defined constants | 2–4 | One-liners. | high |
| 9 | **Riesz representative** — `EuclideanSpace.inner_single_right` + `toDual_symm_apply` | 2–4 | Only works with `open scoped InnerProductSpace` — see dead ends. | med |

## Dead ends
**Catastrophic (never repeat)**
- **Running the linter/auto-fix over the tree: 0.3482 → 0.0429.** Corrupted ≥8 files, destroyed 367
  proofs, cost 4 agent-hours of repair and the run ended at 0.1130. Confidence: certain.
- **Concurrent edits to the same file.** ~40% of all experiment payload was "fixed build errors from
  other agents". Scores went *down* 8 times. Every agent lost work to this.

**Typeclass / elaboration blowups (all reverted to `sorry`, repeatedly)**
- `Lp` type for `EuclideanSpace` — oracle hint #2, confirmed. Use bare `eLpNorm`.
- `essInf`/`essSup` in `Support/MeasureBounds.lean` — heartbeat exhaustion + `IsCoboundedUnder.mk`
  (it's a `def`, not a `structure`; must `unfold` first). Reverted 3×. Nobody tried raising heartbeats.
- `affine_preimage_ball`, `inverse_affine_preimage_unitBall` (BallScaling) — reverted, never retried.
- `isWeakGrad` for add/smul (SobolevSpace/Witnesses) — reverted 2×.
- `Matrix.dotProduct` route in BilinearForm — reverted. Use `matMulE` + `ext` + `simp [Matrix.mulVec_add]`.
- `toDual_symm_apply` without `open scoped InnerProductSpace` — silently fails, broke Energy.lean.
- `integrableOn_Icc_of_Ioo` (StampacchiaTruncation) — reverted at exp046.

**Tactic failures**
- `positivity` on `rpow` goals → use `Real.rpow_nonneg` manually.
- `MulLeftMono ℝ` instance missing → use `one_le_pow₀`.
- `linarith`/`nlinarith` with unsimplified `Nat.find` terms → rewrite explicitly first
  (`rw [h0', pow_zero, div_one]`). Also needs explicit `Nat.cast_nonneg` binding + type annotations.
- `field_simp` without explicit nonzero hypotheses.
- `norm_num` sometimes closes the goal, breaking the tactic after it.

**Mathlib4 renames that broke the build (memorize)**
`div_le_iff`→`div_le_iff₀` · `pow_le_pow_right`→`pow_le_pow_right₀` · `inv_lt_one_of_one_lt`→`…₀` ·
`sum_le_tsum`→`Summable.sum_le_tsum` · `summable_pow_mul_geometric_of_abs_lt_one`→`…_of_norm_lt_one` ·
`summable_geometric_of_lt_one` takes `<` not `≤` · `le_or_lt`→`le_or_gt` · `integral_mul_left`→
`integral_const_mul` · `integral_congr_set`→`setIntegral_congr_set` · `ContDiff.fderiv`→`continuous_fderiv` ·
`EuclideanSpace.measurableEquiv`→`MeasurableEquiv.toLp` · `WithLp.equiv` takes `(Fin d → ℝ)`.
Stale `.olean` files cause phantom errors — delete manually.

## Scaling laws
Single-run observations, not swept. **All confidence: low–medium.**

| Relationship | Observed | Conf |
|---|---|---|
| Aggregate throughput | ~50 sorries/hour across 4 agents ≈ 12.5/agent-hour, sustained 8h | med |
| Agent count vs. useful yield | 4 agents → ~40% of turns were repair, not proof. Effective yield ≈ 2.4 agents | low |
| Score regressions | 8/55 logged experiments decreased the score; 1 was catastrophic (−0.305) | high |
| Build-pass rate | 6/55 experiments logged with `build=1`. 89% of scores are on broken trees | high |
| Difficulty tail | First 0.30 took 8h; the remaining 0.70 is analytic core, not wrappers. Expect ≥10× cost/sorry | med |

| Module | Sorries cleared | Cost | Verdict |
|---|---|---|---|
| MoserIteration/Sequences, CutoffPrep/Profiles | 16, 10 | low | harvest first |
| Oscillation/{BMO,Campanato}, Crossover, WeakHarnack | 9, 7, 5 | low | harvest |
| Localization, BilinearForm, RegularizedEnergy | 4–6 each | low–med | harvest |
| Harnack (arithmetic/geometry only) | ~20 | med | harvest, but conflict-hot |
| Support/MeasureBounds, SobolevSpace/Witnesses, BallScaling | 0 net | high | quarantine |

## Stepping stones
- **`deGiorgi_recurrence_closeout`** (`Y_{n+1} ≤ C·B^n·Y_n^{1+α}`) — oracle hint #4 says it's reusable
  across Moser/DeGiorgi/Harnack. Never actually exploited as a shared engine. Highest-leverage unused asset.
- **`Convex.lineMap_mem` + `dist_lineMap_lineMap`** unlocked the Harnack chain geometry after the
  vadd/vsub approach failed. Generalizes to any chain-of-balls argument.
- **`Real.smoothTransition` toolkit** — the profile machinery is now ~90% proved. The `*_eq_shifted_on_midrange`
  family is a template for the remaining exact-regularization lemmas.
- **`ForwardIteration/Basics` `Nat.find` depth argument** (`exists_forward_iteration_depth`) — the only
  working existence-of-index proof; reuse the `Nat.find_spec`/`Nat.find_min`/`Nat.sub_add_cancel` triple.

## Blind spots (ranked — most promising first)
1. **`#print axioms` audit.** Nobody ever checked whether a "sorry-free" theorem still depends on
   `sorryAx` through imports. The entire score may be inflated. Do this before anything else.
2. **Git commit on every green build + auto-revert on regression.** Would have cost ~zero and saved
   0.235 score. The single highest-value scaffold change.
3. **Module ownership partition** (one agent = one subtree, hard lock). Recovers ~40% of agent time.
4. **Statement-hash guard** — nothing prevented an agent from weakening a theorem statement to close it.
5. **`set_option maxHeartbeats 1000000`** on the MeasureBounds essInf/essSup goals. Never tried; the
   failure was explicitly diagnosed as heartbeat exhaustion and then abandoned.
6. **`exact?` / `apply?` / `aesop` sweep** over all 1212 sorries as a batch pre-pass. Zero attempts logged.
7. **Dependency topological sort** of the sorry graph to find unblocking lemmas instead of leaf-picking.
8. **`~/DeGiorgi-Explained/book/`** — the math exposition was available all run and referenced once.

## Key insight
The score measures **`sorry` tokens removed, not theorems verified** — 89% of logged scores came from
trees that did not compile, and the all-time peak (0.3482) was one of them. Everything gained in 8 hours
was destroyed in 6 minutes by an unguarded linter pass because there was no commit-on-green ratchet.
Progress in this domain is bounded by **artifact durability and write conflicts, not by proof ability.**

## Surprises
- **Expected:** a linter auto-fix is safe cleanup. **Actual:** 0.3482 → 0.0429, 367 proofs destroyed
  across ≥8 files, never recovered (ended 0.1130). **Gap:** auto-fixers were assumed non-destructive and
  were run tree-wide with no snapshot, on files whose proofs depended on exact term structure.
- **Expected:** score is monotone — sorries only go down. **Actual:** it decreased 8 times.
  **Gap:** "append-only progress" assumed no shared mutable state; four agents shared one working tree.
- **Expected:** build status tracks score. **Actual:** the peak score had `build=0`. **Gap:** the harness
  counts `sorry` strings; deleting one and leaving a broken proof scores identically to proving it.
- **Expected:** N agents ≈ N× throughput. **Actual:** ~40% of turns were repairing other agents.
  **Gap:** file-level independence assumed; the module graph is densely coupled and edits raced.
- **Expected:** the hard part is Moser/Harnack analysis. **Actual:** ~all 422 wins were positivity,
  `rfl`, and structural wrappers; the analytic core is untouched. **Gap:** all 1212 sorries were treated
  as equal-weight by the metric. They are not, by orders of magnitude.
- **Expected:** Mathlib is a stable base. **Actual:** ~12 API renames caused repeated build breaks.
  **Gap:** agents wrote from pretrained Mathlib memory instead of checking the installed version.
- **Expected:** the blackboard tracks the best score. **Actual:** `Current Best: 0.0000` for the entire
  run. **Gap:** nobody owned the header; every agent appended and none updated shared state.

## Devil's advocate
The best score is **inflated, and the headline number is invalid.**
1. **0.3482 is not a real result.** `build=0`. Lean never checked those files. A deleted `sorry` with a
   non-compiling proof under it decrements the counter exactly as much as a real proof. Discard it.
2. **Even 0.3102 (build pass) is unaudited.** No `#print axioms` was ever run. A file can be textually
   sorry-free while every theorem in it transitively depends on `sorryAx` from an import. Until that
   check runs, treat 0.3102 as an *upper* bound on verified content, not a measurement.
3. **The metric is uniform over wildly non-uniform work.** 422 sorries closed, and essentially all of
   them are `X_pos`, `X_nonneg`, `rfl`, and `IsSolution.neg_ball`. `deGiorgi_recurrence_closeout`, the
   Moser iteration closeout, and Harnack itself remain open. 35% of the count is maybe 5% of the math.
   A reader seeing "35% of De Giorgi–Nash–Moser formalized" would be badly misled.
4. **No statement-integrity guard.** Agents freely edited theorem files. Nothing detected a hypothesis
   added or a conclusion weakened to make a goal closeable. This is the classic sorry-free-but-vacuous
   failure and it was structurally undetectable in this run.
5. **Per-agent attribution is noise.** Scores were sampled by separate harness runs against a shared,
   concurrently-mutating tree ("running second harness for latest changes"). Two agents logged 0.3102
   at the same minute for different work. Do not trust any single agent's delta.
6. **What IS solid:** the *technique* list. The Mathlib rename table, the `smoothTransition.contDiff.comp`
   pattern, `Convex.lineMap_mem`, and the quarantine list are grounded in repeated compile-verified
   observation and will transfer. Trust the methods; distrust the number.

## Experiment order
1. **Audit before producing.** Reproduce the 0.3102 green build. Run `#print axioms` on every top-level
   theorem. Log the *verified* baseline. If it's below 0.3102, that gap is the real finding. (~1 exp)
2. **Install the ratchet.** `git commit` on every green build; auto-revert any change that increases the
   sorry count or breaks the build. **Ban tree-wide linter/auto-fix runs.** (~1 exp, non-negotiable)
3. **Partition.** Assign one agent per subtree, hard-locked. No cross-subtree edits; file a request on
   the blackboard instead. Expected +40% effective throughput. (~0 exp)
4. **Batch-harvest tier 1–5** from *What works*, one agent per module, in the order of the module table.
   Reproduce ~0.31 in an estimated 2–3h instead of 8. (~10 exp)
5. **Attack the quarantine list with new tools** — `maxHeartbeats` on MeasureBounds, `exact?`/`aesop`
   sweep on the residual. This is where the untried leverage is. (~5 exp)
6. **Only then** touch the analytic core, starting from `deGiorgi_recurrence_closeout` as a shared engine.
   Everything before this step is bookkeeping; this is the actual benchmark.
```

Two things I'd flag as judgment calls in there: I ranked the verified 0.3102 as the "winning recipe" over the nominal peak 0.3482, and the Scaling laws section is honest that this domain has no swept hyperparameters — the tables are observed rates from one run, marked low/medium confidence, rather than invented relationships.

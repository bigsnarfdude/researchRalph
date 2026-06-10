# MISTAKES.md — Tactics that failed and why

## 2026-03-27 — agent0

1. **Shotgun tactic search on complex problems**: The `first | solve | tactic1 | solve | tactic2 | ...` pattern fails on any problem requiring multi-step reasoning (Equiv chaining, partial fractions, modular arithmetic). Single tactics can't solve problems needing intermediate lemmas.

2. **rpow_natCast rewrite on ℝ**: Tried `rw [rpow_natCast]` to convert `10 ^ (2 : ℝ)` to `10 ^ (2 : ℕ)` but the pattern didn't match because the exponent was already a literal `2` with ambiguous type. Solution: use `show ... from by norm_num` to explicitly rewrite the base.

3. **Equiv.right_inv with wrong argument**: For `σ : Equiv ℝ ℝ` with `h₂ : σ.2 3 = 9`, to get `σ.1 9 = 3`, I initially tried `σ.right_inv 9` and rewrote with h₂. But `σ.right_inv 9` gives `σ.1 (σ.2 9) = 9`, not what I need. Correct: use `σ.right_inv 3` which gives `σ.1 (σ.2 3) = 3`, then rewrite `σ.2 3` to `9` using h₂.

4. **nlinarith insufficient for division reasoning**: For proving `(x+y)² ≥ 16` from `(x+y)²(xy-1) = 4(xy)²` and `xy > 1`, nlinarith can't combine division/cancellation. Need explicit intermediate steps: show `4(xy)² ≥ 16(xy-1)` via `(xy-2)² ≥ 0`, then use `mul_le_mul_right` to cancel `(xy-1) > 0`.

5. **Nat.cast coercion in linarith**: After `Real.log_pow`, coefficients appear as `↑3` (Nat.cast) which `linarith` can't work with. Fix: `push_cast` before `linarith`.

6. **Linter auto-modifying proof files**: The project linter aggressively simplifies proof files, often replacing working proofs with non-compiling ones. This caused 28 regressions in exp015. Must use `bash` directly to write files, or restore from exp014 after linter runs.

## 2026-03-27 — agent1

7. **omega cannot prove quadratic ℕ identities**: Tried `omega` for `n^2 + 2 - 3*n = (n-1)*(n-2)` over ℕ. omega is linear-only; ℕ subtraction truncation makes it worse. Fix: `zify [bounds]; ring`.

8. **nlinarith insufficient for ℕ multiplicative cancellation**: Tried `nlinarith` for `n-2 = (n-1)*(n-2) → False` with n ≥ 4. Fix: use `mul_right_cancel₀` (not `Nat.eq_of_mul_eq_right` which doesn't exist).

9. **Nat.modEq_iff_dvd' argument order**: Requires `a ≤ b` for `a ≡ b [MOD n]`. For `171 ≡ 80`, need `.symm` first to get `80 ≡ 171`, then apply with `80 ≤ 171`.

10. **Overwriting working proofs in merge**: Copying all proofs into a merged experiment (including ones already solved differently) caused net regressions. Only copy proofs for problems that FAIL in the target.

## 2026-03-27 — agent0 (continued)

7. **Bash heredoc semicolon issue**: When writing Lean proofs via bash heredoc, `have : ... := by ring; rw [h] at this` on one line makes `ring` close the inner goal, then `rw` runs as a separate tactic in the outer context and fails. Must put `have ... := by ring` on its own line, followed by `rw [h] at this` on the next line.

8. **div_lt_div_of_pos_left argument order**: `div_lt_div_of_pos_left (ha : 0 < a) (hb : 0 < b) (h : b < c)` gives `a / c < a / b` (smaller denominator = larger fraction). The conclusion direction is counterintuitive — it gives `a/c < a/b`, not `a/b > a/c`. Need to match the goal direction.

9. **omega cannot reason about % with variable modulus**: `omega` can handle `a % n = b % n` when n is a literal, but NOT when n is a variable. For `171 % n = 80 % n → n | 91`, must use `Nat.modEq_iff_dvd'` instead of omega.

10. **Finset.sum_nbij with ℕ subtraction functions**: Using `Finset.sum_nbij (fun k => k - 1)` for variable substitution in sums fails because omega can't simplify `(fun k => k-1) (j+1)` to `j` through the coercion layer. Need explicit `obtain ⟨j, rfl⟩` destructuring instead.

11. **Binomial identity via Finset manipulation is extremely tedious**: The identity ∑k*C(n,k) = n*2^(n-1) requires the absorption identity, variable substitution in Finset.Icc, and Nat.sum_range_choose. Each step involves fighting with Finset API, omega limitations, and ℕ subtraction. Would be much easier with a higher-level Mathlib lemma if one exists.

## 2026-03-27 — agent1 (continued)

12. **ring cannot handle Nat.cast atoms in ℚ**: `ring` over ℚ with `↑n` (Nat.cast) as a free variable doesn't work when `↑(n+k)` appears as independent atoms. Must first rewrite `↑(n+k)` to `↑n + k` via `rw [show (↑(n+k) : ℚ) = ↑n + k from by push_cast; ring]`.

13. **field_simp incomplete on nested fractions a/b / (c/d)**: Must compute inner denominator as explicit `have`, rewrite, then `field_simp` on simpler form. Two-pass approach: `field_simp [h1, h2]; ring` for the inner denominator, then `field_simp [h1, h2, h3, h_denom_ne]` for the outer division.

## 2026-03-28 — agent3

14. **aime_1984_p5 is unprovable under Lean's Real.log**: Lean defines `Real.log x = log|x|` for `x ≠ 0` (confirmed via `Real.log_eq_zero : log x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1`). This means `logb 8 (-64) = logb 8 64 = 2`. Counterexample: a=-64, b=8 satisfies both h₀ and h₁ but ab=-512≠512. Spent 30+ minutes before discovering this.

15. **aime_1988_p3 is unprovable**: x=1 satisfies h₀ (0<1) and h₁ (logb2(logb8(1))=logb8(logb2(1))=0=0) but the conclusion (logb2(1))²=0≠27 is false.

16. **Tried wlog tactic for g_inj**: `wlog hle : f r₁ ≤ f r₂ with H` failed because H had the wrong type. Had to use manual `suffices aux` + `rcases le_total` instead.

17. **Bash heredoc cp -r creates nested directory**: `cp -r exp017 exp048` when exp048 doesn't exist creates exp048/exp017/ (nested), not exp048/ with contents. Need to flatten afterwards.

## 2026-03-28 — agent1 (session 2)

14. **exact_mod_cast with Finset sums over ℤ→ℝ**: `exact_mod_cast (h : ∑ k, k = n)` does NOT close `∑ k, (↑k:ℝ) = n` because the goal has `↑k` (cast applied to each element) while hypothesis has `k` (no cast). Need `Int.cast_sum` first to convert `∑ ↑k = ↑(∑ k)`.

15. **push_cast + simp_all creates spurious disjunctions**: When `x : ℝ` is a free variable and `push_cast` normalizes expressions, subsequent `simp_all` can generate goals like `∑ k = 3570 ∨ x = 0`. Avoid mixing `push_cast` with `simp_all` when free variables are present.

16. **abs_add doesn't exist**: The triangle inequality `|a+b| ≤ |a|+|b|` is `abs_add_le`, not `abs_add`. Also `abs_sum_le_sum_abs` is in `Finset` namespace: `Finset.abs_sum_le_sum_abs`.

17. **field_simp + linarith fails with division after clearing**: After `field_simp`, if the goal still contains `a/b` terms (from incomplete clearing), `linarith` can't handle them. The fix is to use `nlinarith` with SOS witnesses, or restructure the proof to avoid residual divisions. For trig identities, it's better to rewrite with `sin_two_mul` + `cos_two_mul` BEFORE field_simp, so the denominators become products of sin/cos (no trig functions of composite arguments).

## 2026-03-28 — agent4

18. **omega fails on Nat.cast abs expressions**: After `interval_cases a` substitutes ℕ literal for `a`, the goal `|↑7 - 22| = 15` (where `↑` is Nat.cast to ℤ) fails with omega because omega doesn't evaluate `|(↑7 : ℤ) - 22|`. Fix: use `simp_all` as fallback after `omega`, which normalizes the cast and evaluates abs.

19. **Built exp058 from wrong base (exp052_merge instead of exp048)**: exp052_merge had weaker proofs for 45 files compared to exp048. Lost ~2 hours debugging before discovering the directories diverged. Always check which directory actually produced the best score.

20. **Sorry files pass evaluation silently**: Files containing `sorry` compile with exit code 0 (just a warning). The run.sh evaluator counts them as PASS. Found 4 sorry files inflating exp048's score: aime_1984_p5, amc12a_2010_p22, imo_1966_p4, mathd_numbertheory_126.

21. **rpow_mul needs `≤` not `<` for positivity**: `rpow_mul` requires `0 ≤ x`, not `0 < x`. Use `le_of_lt` if you have strict positivity, or establish `≤` directly with `linarith`.

## 2026-03-28 — agent1 (session 3)

22. **Built exp058 from exp057 initially instead of exp017**: exp057 had 67 failures (only shotgun proofs), while exp017 had all custom proofs. Wasted time debugging before discovering the base was wrong. Always verify custom proofs are present after copying.

23. **Nat.pow_left_injective takes `n ≠ 0` not `1 ≤ n`**: Expected `1 ≤ 2` but it needs `2 ≠ 0` (as `two_ne_zero`). Also `pow_left_injective` is an Injective function, not a direct equality proof — need to apply it to the right argument order.

24. **Nat.div_mul_cancel gives `a/b * b = a` not `b * (a/b) = a`**: Multiplication order matters for ℕ. Use the `.symm` if needed, or adjust variable names to match.

## 2026-03-28 — agent5

25. **omega can't handle a^2 % k**: omega only does linear arithmetic, so `a^2 % 3 = 0 ∨ a^2 % 3 = 1` requires case-splitting on `a % 3` first, then `rw [Int.mul_emod, h]; simp`.

26. **ring_nf + linear_combination coefficient drift for ℂ**: After `ring_nf`, the normal form changes between calls, making `linear_combination c * hi` fail because `c` must be exact. Alternative: use `Complex.ext` then `simp [...] <;> ring` to work on re/im parts separately.

27. **omega can't prove non-primality**: After `interval_cases m`, goals like `¬ Nat.Prime 4` can't be closed by omega. Use `norm_num` or `decide` instead.

28. **Concurrent agents clobbering shared experiment directory**: Multiple agents writing to attempts/exp058 simultaneously caused race conditions. Solution: create isolated copies (exp058_a5) for scoring.

## 2026-03-28 — agent4 (continued)

22. **Built exp058 from wrong directory initially**: Copied exp052_merge (which had weaker proofs) instead of exp048 (which had the actual best score). Cost ~2 hours of debugging when eval showed many failures.

23. **Forgot to check sorry files when merging**: exp048 had 4 sorry files silently passing eval. Didn't discover this until manually checking each file.

24. **Tried to fix sorry files with `omega` but wrote to wrong directory**: The linter may have reverted changes, or the write went to a different copy. Always verify after writing.

## 2026-03-28 — agent6

18. **Finset.sum_sub_distrib doesn't exist (or doesn't apply)**: Tried to split ∑(kx-1) into ∑k*x - ∑1. The correct approach in Lean is `simp only [Finset.sum_sub_distrib]` but it failed with "max recursion depth". Better to use `simp only [mul_sub, Finset.sum_sub_distrib]` or work with explicit `Finset.sum_mul`.

19. **abs_sub_abs_le_abs_sub is wrong direction**: `||a|-|b|| ≤ |a-b|`, not `|a-b| ≤ |a|+|b|`. For the reverse triangle inequality (`|a-b| ≤ |a|+|b|`), use `abs_sub_le a b` which gives `|a-b| ≤ |a|+|b|` (wait, actually need to derive from `abs_add` applied to a and (-b)).

20. **List.Pairwise extraction**: `simp [List.pairwise_cons]` doesn't always destructure cleanly. Better pattern: `rw [List.pairwise_cons]; obtain ⟨hall, htl⟩ := h₈; apply hall; simp` for each pair.

21. **nlinarith timeout on polynomial systems**: nlinarith with many polynomial hypotheses (6+ cubic equations) times out trying to find witnesses. Must provide intermediate algebraic steps (like computing r, s explicitly) rather than asking nlinarith to derive the final answer from raw equations.

25. **Almost wasted time on mathd_algebra_282**: Initially thought it was solvable (8^(1/3) = 2), but Lean's ℕ division makes 1/3 = 0, so 8^(1/3) = 1. The problem asks for total 79 but correct value is 78. Verified with `norm_num` that `(8:ℝ)^(1/3) = 1`.

## 2026-03-28 — agent7

29. **Built from corrupted exp052_merge base**: Started by copying exp052_merge which had tactic sweep proofs instead of custom proofs. Wasted significant time writing "new" proofs that were already solved in exp017. The exp052_merge directory scored 0.6721 vs exp017's 0.8279.

30. **Wrote 43 "new" proofs that were all already solved**: All SOS inequalities, induction proofs, ring identities, and AMC problems I proved were already passing in exp017's tactic sweep. None of the 43 proofs targeted actually-failing problems.

31. **Finset reindexing is extremely painful**: Trying to prove ∑k*C(n,k)=n*2^(n-1) required reindexing Icc 1 n → range n, which involved sum_nbij/sum_map with ℕ subtraction that omega can't handle. Abandoned after 4 attempts.

## 2026-03-28 — agent6 (session 2)

26. **Finset.sum_sub_distrib causes max recursion on large sets**: Using `Finset.sum_sub_distrib` directly on `Finset.Icc 1 84` exceeds maxRecDepth with default settings. Fix: rewrite `(kx-1)` as `(kx+(-1))` first using a helper lemma, then use `Finset.sum_add_distrib` which doesn't trigger the recursion issue.

27. **abs_add doesn't exist in current Mathlib**: The triangle inequality is `abs_add_le`, not `abs_add`. Spent time debugging unknown identifier errors.

28. **gen_proofs script overwrote my custom proof**: The automated gen_proofs_v11.py script wrote shotgun-tactic proofs to exp105, overwriting my hand-crafted amc12a_2010_p22 proof. Must either use a separate output directory for automation or protect custom proofs from overwrite.

## 2026-03-28 — agent2

32. **Tried nlinarith with SOS for imo_2006_p3**: The bound involves 9√2/32 which makes the problem non-polynomial. nlinarith with any set of sq_nonneg witnesses will fail because the tight constant is irrational. Need a fundamentally different proof technique (e.g., Lagrange multipliers, substitution to eliminate √2, or Schur-type inequality).

33. **Background eval used relative paths**: First attempt at evaluating exp079 used relative paths in `lake env lean`, causing all 244 problems to fail. Must use absolute paths when running from outside the miniF2F-lean4 directory.

34. **gen_proofs_v12.py too slow under load**: The automated tactic search script runs sequentially through problems, each trying 100+ tactics with 120s timeout. With 21 concurrent lean processes, each check takes 2-3 minutes. Total estimated time: 12+ hours. Should have used parallel evaluation within each problem.

## 2026-03-28 — agent7 (session 2)

32. **Automated tactic search on hard problems**: Ran 223 tactic combinations across imo_1990_p3, imo_2006_p3, imo_1993_p5, aime_1987_p8. Zero passes. These remaining problems are genuinely hard and can't be solved by any single or two-step tactic combination. Need multi-step custom proofs.

33. **imo_1990_p3 bound approach**: Tried to bound n ≤ some value then interval_cases. The bound n² ≤ 2^n+1 is trivially true for all large n (2^n >> n²), so this gives no upper bound. Need number-theoretic argument (order of 2 mod p, LTE).

34. **div_lt_div_iff vs div_lt_div_iff₀**: In current Mathlib, `div_lt_div_iff` doesn't exist. Use `div_lt_div_iff₀` which requires CommGroupWithZero. Also `lt_div_iff₀` exists for a < b/c form.

35. **abs_sub_abs_le_abs_sub has wrong type**: `abs_sub_abs_le_abs_sub S₁ S₂` gives `|S₁| - |S₂| ≤ |S₁ - S₂|`, not `|S₁ - S₂| ≤ |S₁| + |S₂|`. For the latter, use `abs_sub S₁ S₂`.

36. **push_cast at ℕ inequalities doesn't auto-cast for linarith**: After `push_cast`, ℕ hypotheses stay in ℕ. Need `exact_mod_cast` to explicitly cast `6*n < 7*k` from ℕ to `(6:ℝ)*↑n < 7*↑k` before `linarith` can use them with ℝ goals.

## 2026-03-28 — agent3 (session 2)

18. **Tried k₀+1 approach for IsGreatest upper bound**: For aime_1987_p8, initially tried to show k₀+1 always works when m≥113. But for m=113, k₀=98: k₀+1=99 exceeds the upper bound (8·99=792>791=7·113). The correct approach is to use k₁=6m/7+1 and k₁+1 as witnesses — omega can verify these satisfy the ℕ bounds via integer division properties.

19. **exact_mod_cast fails on mul_comm mismatch**: `div_lt_div_iff₀` gives `a*d < c*b` (not `a*d < b*c`). After omega proves `8*(m+k) < 15*m`, casting gives `8*(m+k) < 15*m` but the goal has `8*(↑m+↑k) < ↑m*15` (reversed multiplication). Fix: use `linarith` after `exact_mod_cast` to handle commutativity.

20. **System load causes massive eval failures**: With LA=15+, the 60s timeout in run.sh causes ~210/244 proofs to fail (only 34 pass). Even at 120s, most proofs time out under heavy concurrent load. Need to wait for LA<3 to get accurate scores.

## 2026-03-28 — agent0 (session 2)

32. **Tried to prove imo_2006_p3 via nlinarith SOS**: The 9√2/32 constant involves √2 which nlinarith can't handle (polynomial solver only). Would need to square both sides first, losing the √2, but the result is degree-8 which nlinarith still can't crack without astronomical witness terms.

33. **Tried Complex.ext for complex arithmetic**: `Complex.ext` requires goals of the form `a = b` where `a,b : ℂ`. For identities like `(1+I)^4 = -4`, need to use `apply Complex.ext` then prove re and im parts separately. `ring` doesn't know I^2 = -1.

34. **Spent time on problems already solved in other experiments**: Before writing any proof, ALWAYS check all experiment directories for existing passing proofs. Three of the "remaining 23" problems were already solved in exp058.

35. **Launched 2 scoring runs on same directory**: Accidentally launched run.sh for exp106_a2 twice (once with 1 proof, once with 2). The second run scores a directory that was being modified by the first. Should only launch one scoring instance per directory.

37. **aime_1991_p6 Finset.Icc rewrite after set**: After `set S := ∑ k ∈ Finset.Icc 19 91, ...`, trying to `rw [show Finset.Icc 19 91 = ...]` at hS35 fails because S has been folded. Need to unfold S or use `change` or work directly without the `set`.

38. **Int.floor_le_floor type mismatch**: `Int.floor_le_floor` expects `a ≤ b` (ℝ) but provides `⌊a⌋ ≤ ⌊b⌋` (ℤ). The direction is: `a ≤ b → ⌊a⌋ ≤ ⌊b⌋`. Make sure to provide the ℝ inequality as input.

## 2026-03-28 — agent3 (session 2, continued)

21. **omega can't convert ℤ Finset sum to ℕ**: `∑ ↑f = c : ℤ` where f : ℕ → ℕ. omega doesn't know `∑ ↑f = ↑(∑ f)`. Use `exact_mod_cast` instead.

36. **gen_proofs_v12.py died under load**: The automated tactic search script was killed (OOM or timeout) due to 24+ concurrent lean processes. The script needs to wait for idle system or be more aggressive about parallelism/timeout management.

39. **Finset.Icc type inference failures**: `Finset.Icc 19 57` without type annotation fails to synthesize `LocallyFiniteOrder`. Always write `Finset.Icc (19:ℕ) 57`.

40. **ext on Finset causes max recursion**: `ext; simp [Finset.mem_Icc, Finset.mem_union]; omega` for proving Finset equality hits recursion depth. Use `decide` for concrete Finsets or explicit `Finset.Icc_union_Icc_eq_Icc`.

41. **ring_nf changes division to multiplication**: `ring_nf` transforms `↑k / 100` to `↑k * (1/100)`, breaking subsequent `rw [Int.floor_add_intCast]` pattern matching. Use `show ... from by ring` instead.

37. **gen_proofs_v12.py killed with SIGTERM (exit 144)**: The automated tactic search script was killed before producing any output. Under LA 12+ with 25 concurrent lean processes, even Phase 1 (checking all 244 files) couldn't complete. The script needs: (a) sequential execution (not parallel) when load is high, (b) a "wait for idle" mechanism, or (c) shorter per-file timeouts with retry logic.

## 2026-03-28 — agent5

16. **Merging under system load causes score regression**: My exp110 merge (exp079+3 proofs) scored 0.8934 instead of expected 0.9098 because 11+ concurrent lean processes caused timeout failures on 4-5 proofs. Always wait for idle system or increase timeouts.

17. **imo_1990_p3 cannot be bounded by size arguments**: Tried to show n ≤ K for finite K, but 2^n grows faster than n² for all n ≥ 2. The upper bound requires number theory (order of 2 mod n, p-adic valuation), not size.

18. **amc12a_2009_p25 periodicity requires 15+ recurrence steps**: The sequence a(n) with tangent-addition recurrence has period 24 (anti-period 12). Proving a(2009)=0 requires either: (a) computing 15 recurrence steps with √3 arithmetic, or (b) proving the anti-period property via Fibonacci analysis. Both are very tedious in Lean.

## Agent4 Session - 2026-03-28

1. **positivity fails on (0:ℝ) < ↑n + ↑k when n,k are ℕ** — need to cast first or use `linarith [Nat.cast_nonneg k]` with a positivity proof on n
2. **Int.floor_eq_iff signature changed** — no longer takes positivity proof as first arg; just call `Int.floor_eq_iff` directly and prove the two sides
3. **`simp only [h₀ _ _ h1]` fails when h₀ has explicit binders** — use `rw [h₀ _ _ h1]` instead
4. **Shell `| solve |` detection missed multiline grep** — use Python for complex content analysis, not bash one-liners
5. **All 244 files passing `lake env lean` was misleading** — Lean was warm/cached; re-running cold shows actual failures. Template proofs (`first | solve | ...`) fail compilation but only after all tactics are tried (takes time)
6. **Trying to bound imo_1990_p3 without order theory** — wasted 30+ minutes. This problem fundamentally requires proving n is a power of 3 using multiplicative orders modulo primes. No computational shortcut works.
7. **SOS witnesses for imo_2006_p3** — `nlinarith` with 15+ SOS terms still fails. The 9√2/32 constant requires an exact SOS decomposition involving √2, which `nlinarith` can't handle.

## 2026-03-28 — agent0 (session 2, continued)

35. **nsmul not recognized by omega**: After `simp [Finset.sum_const]`, the hypothesis has `73 • n` (nsmul) instead of `73 * n`. omega can't handle nsmul directly — need `simp [nsmul_eq_mul]` first.

36. **Int.floor_lt gives cast confusion**: `Int.floor_lt.mpr (show x < 2 from ...)` where `2` is inferred as `ℝ` fails because `Int.floor_lt` needs the RHS as `(m:ℤ)`. Use `show x < (2:ℤ) from by push_cast; linarith` instead.

## 2026-03-28 — agent1 (session 4)

32. **div_lt_div_iff is now div_lt_div_iff₀**: In current Mathlib, `div_lt_div_iff` doesn't exist. Use `div_lt_div_iff₀` instead (with the ₀ suffix). The signature: `div_lt_div_iff₀ (hb : 0 < b) (hd : 0 < d) : a / b < c / d ↔ a * d < c * b`.

33. **push_cast + omega doesn't work for ℝ inequalities**: After `push_cast` on ℝ-valued inequalities containing ℕ casts, `omega` can't close the goal because the atoms are still ℝ. Must use `exact_mod_cast` to convert back to ℕ first, or `linarith` for ℝ.

34. **Semicolons in `have ... := by ring; rw [...]`**: The semicolon makes `rw` part of the `have` proof, not a separate tactic. When `ring` closes the `have` goal, `rw` sees "no goals" and fails. Use multi-line syntax instead.

## 2026-03-28 — agent7 (session 3)

25. **field_simp sometimes closes goals unexpectedly**: When using `field_simp [hs_ne, hsm1_ne]` inside a `show ... from by ...` clause, `field_simp` may completely close the goal, leaving subsequent tactics like `ring` or `nlinarith` with "No goals to be solved". Fix: test `field_simp` alone first, then add `ring`/`nlinarith` only if needed.

26. **Inlining Mathlib Archive proofs has compatibility issues**: Tried to inline the imo_1988_p6 Vieta jumping proof from Mathlib's Archive. Multiple API changes between Lean versions caused ~10 errors (ambiguous names like `ne_of_lt`, changed `Set.mem_image_of_mem` signature, etc.). Not worth debugging for one problem.

27. **Background scoring can be invalidated by file changes**: Started background scoring of exp142, then added a new proof file. The scoring job scored the OLD version. Must either wait for scoring to finish or restart the job after changes.

## 2026-03-28 — agent6 (session 3)

29. **nlinarith can't handle 1/s terms after div_eq_iff**: After `rw [div_eq_iff d]`, the goal has `1/s` atoms. nlinarith can't reason about these (nonlinear in s). Fix: add `field_simp [sn]` between `div_eq_iff` and `nlinarith` to clear the 1/s terms first.

30. **neg_mul_neg rewrite fails on `-1 * (-(s-1)/(s+1))`**: The pattern `- ?a * - ?b` doesn't match `-1 * (-(s-1)/(s+1))` because `-1` isn't syntactically `- ?a`. Fix: use `have : -1 * (-(s-1)/(s+1)) = (s-1)/(s+1) := by ring` then rewrite, or use `field_simp; ring` to normalize.

31. **Semicolons between `have` declarations cause "No goals to be solved"**: `have h1 := by nlinarith; have h2 := by linarith` treats the semicolon as a tactic separator within h1's proof, not as separating two `have` declarations. The `have h2` runs in h1's context where the goal is already solved. Fix: use newlines, not semicolons, between `have` declarations.

32. **amc12a_2009_p25 requires 150+ lines**: The tangent recurrence period-24 proof needs computing 20 recurrence steps through √3 algebra (a(5)=0, a(6)..a(17)=0, a(18)..a(26)=a(2)). Each step is 3-4 lines of field_simp + nlinarith. No shortcut found — the brute-force computation is the only viable approach.

## 2026-03-28 — agent5 (session 2)

29. **field_simp generates unresolvable goals with nested fractions**: Using `field_simp` on expressions like `(1+s)/(1-s) + (s + (1+s)/(1-s))/(1-s*(1+s)/(1-s))` leaves residual inverses that `nlinarith` can't handle. The product-of-denominators technique (multiply by D, show D≠0) is much more reliable.

30. **Spent time re-proving amc12a_2009_p25 when agents 6,7 already had it**: Should have checked the blackboard and other experiment directories more carefully before investing effort. The proof was independently discovered by agents 2, 3, 6, and 7.

## 2026-03-30 — agent1

31. **push_cast [hd] fails for ℕ division wrapper**: In imo_1988_p6, `push_cast [hd]` where `hd : d^2 = (a^2+b^2)/(a*b+1)` in ℕ doesn't properly convert ℕ division to ℝ multiplication. Fix: use `Nat.div_mul_cancel` to get the multiplicative form `d^2*(a*b+1) = a^2+b^2`, then `exact_mod_cast`.

## 2026-03-30 — agent7 (session 4)

37. **List.Chain renamed to List.IsChain in current Mathlib**: `List.Chain.tail`, `List.Chain.rel_head`, etc. are now `List.IsChain.tail`, `List.IsChain.rel_head?`, `List.IsChain.cons'`. The old names are deprecated with a warning but some don't exist at all (e.g., `Chain.rel_head` → use `IsChain.rel_head?`).

38. **isChain_cons' rewrite fails on (a :: l ++ [0])**: When the target is `IsChain R (a :: l ++ [0])`, `rw [isChain_cons']` can't match the pattern `IsChain R (x :: l)` due to the `l ++ [0]` not unifying with a single variable. Fix: extract manually using `hl.tail` for the tail and `hl.rel_head?` for the head relation.

39. **simp at hlt before case split vs after**: `simp only [map_nil, sum_nil] at hlt` on `(l₂.map fib).sum` while l₂ is a variable keeps `(l₂.map fib).sum`. After `cases l₂ with nil`, the term becomes `(map fib []).sum` which omega treats as opaque. Fix: use `simp at hlt` AFTER the case split, not before.

40. **convert ... using 2 can overshoot**: `convert key using 2` may close more goals than expected, making subsequent `congr 1; rfl` fail with "No goals to be solved". Check if `convert` alone suffices.

41. **Scoring under load gives false negatives**: With LA=25+ and 14 concurrent run.sh jobs, the 60s timeout causes 3 false failures. All 3 new proofs pass individually at 120s. Must score on idle system.

42. **Greek letter φ not valid Lean 4 identifier**: Use ASCII names like `golden` instead.

32. **Tried interval_cases + omega for mathd_numbertheory_126**: After showing x+3 | 40 and bounding x ≤ 37, `interval_cases x <;> omega` fails because omega can't handle Nat.gcd/Nat.lcm constraints. Also discovered the problem is unprovable (counterexample x=37, a=1480).

33. **imo_1987_p6 duplicate h₀ makes it unprovable**: The second hypothesis `h₀ : ∀ k, k ≤ ⌊√(p/3)⌋ → Prime(f k)` shadows the first `h₀ : ∀ x, f x = x²+x+p`. Without the function definition, the conclusion requires primality for i up to p-2 but h₀ only provides it up to ⌊√(p/3)⌋, which is much smaller.

34. **Attempted golden ratio floor formula for imo_1993_p5**: f(n) = ⌊(n+1)φ⌋-1 where φ = (1+√5)/2 satisfies all properties. But the proof requires: sqrt bounds, floor arithmetic identity ⌊⌊mφ⌋·φ⌋ = ⌊mφ⌋+m-1, and irrationality of φ. Estimated 80+ lines of Lean. Abandoned for now.

35. **native_decide on Finset.range 18M is too slow**: Agent6's amc12a_2020_p21 approach using range 18144001 won't terminate. Fix: show n | 18144000 and use Nat.divisors (only 360 elements) instead. native_decide on 360 divisors completes instantly.

## 2026-03-30 — agent3

42. **Wrote 7 custom proofs for problems already passing**: Spent time writing custom proofs for imo_1961_p1, imo_1964_p1_1/2, imo_1973_p3, imo_1974_p5, imo_1984_p2, amc12a_2002_p12 — all were already passing with shotgun tactic proofs in exp142_a3. Should have run targeted eval on these problems FIRST before investing proof-writing effort.

43. **Assumed shotgun proofs can't solve hard problems**: The shotgun tactic cascade (`first | solve | linarith | ... | nlinarith [sq_nonneg _] | ...`) is more capable than expected, solving IMO problems like imo_1973_p3 and imo_1984_p2 via nlinarith with SOS witnesses.

## 2026-03-30 — agent4

29. **mathd_numbertheory_126 is unprovable**: Counterexample (x=37, a=1480) satisfies all hypotheses (gcd(1480,40)=40=37+3, lcm(1480,40)=1480=37*40) but a≠8. The problem has two valid (x,a) pairs: (5,8) and (37,1480). The "smallest a" condition h₃ doesn't disambiguate because each pair has a unique a.

30. **amc12a_2020_p13 is unprovable due to ℕ division**: All exponents (1/a, 1/b, 1/c) are ℕ division, giving 0 for a,b,c ≥ 2. Both sides evaluate to 1. Can't derive b=3 from 1=1.

31. **mathd_algebra_282 is unprovable**: 8^(1/3) in the goal uses ℕ division: (1:ℕ)/(3:ℕ) = 0, so 8^0 = 1. Sum = 1+9+64+4 = 78 ≠ 79.

32. **Tried to formalize imo_1990_p3 from scratch — underestimated complexity**: The order argument (minFac(n)=3) requires ~100 lines of Lean with ZMod machinery. The v_3 analysis using LTE adds another ~50 lines. Much harder than expected.

## 2026-03-30 — agent6

33. **Assumed Real.log returns 0 for negatives**: Lean's `Real.log x = expOrderIso.symm ⟨|x|, ...⟩` — it uses absolute value, NOT 0, for negative inputs. So `logb 8 (-64) = logb 8 64 = 2`, not 0. This means aime_1984_p5 is broken differently than expected (counterexample a=-64, b=8 gives a*b=-512≠512).

34. **Tried `push_cast [hd]; ring` for ℕ→ℝ in imo_1988_p6**: The `hd` is about ℕ division `d²=(a²+b²)/(ab+1)` which doesn't push_cast cleanly. Fix: first prove the multiplicative form `d²*(ab+1)=a²+b²` via `Nat.div_mul_cancel`, then `exact_mod_cast`.

35. **interval_cases without explicit upper bound**: `interval_cases n` fails if Lean can't find an upper bound in context. Need `have : n ≤ 37 := by omega` first.

36. **Assumed imo_1987_p6 is broken due to h₀ shadowing**: The shadowed hypothesis IS accessible via `assumption` or `show ... from by assumption`. Lean 4 doesn't remove shadowed hypotheses from context — they're just not reachable by name.

37. **Overly complex approach to amc12a_2020_p21**: Tried prime factorization analysis first. The simpler approach: n ≤ lcm(120,n) = 5*gcd(3628800,n) ≤ 5*3628800 gives bound, then native_decide.

## 2026-03-30 — agent3 (continued)

44. **Launched scoring under high load (LA=24)**: Multiple agents running concurrent evaluations caused load to spike to 24+, making scoring unreliable. Should check `uptime` before launching scoring and wait for LA < 3.

45. **Tried to formalize imo_1978_p5 without rearrangement inequality**: The proof requires the rearrangement inequality (Mathlib has Monovary/Chebyshev but not the exact form needed). Abandoned after recognizing the gap.

46. **Attempted imo_1993_p5 with Zeckendorf**: Mathlib has Zeckendorf but the function was `noncomputable`, blocking `native_decide`. Would need 100+ lines for a constructive proof.

## 2026-03-30 — agent0 (session 3)

39. **linarith in ZMod p**: ZMod p has no linear order, so `linarith` fails on any goal or hypothesis in ZMod p. Must use `linear_combination` for ring arithmetic or `norm_cast` + ℕ/ℤ reasoning. Wasted ~30 min debugging cascading linarith failures.

40. **push_cast in ZMod p**: `push_cast` sometimes creates type mismatches between `(2 : ZMod p)` and `((2 : ℕ) : ZMod p)`. Fix: use `show (2 : ZMod p) = ((2 : ℕ) : ZMod p) from by push_cast; ring` for explicit cast, then `ZMod.natCast_eq_zero_iff`.

41. **orderOf_pos needs LeftCancelMonoid**: `orderOf_pos (2 : ZMod p)` fails because ZMod p isn't a LeftCancelMonoid (it's a ring, not a group under multiplication). Work around: prove `0 < d` from `d | 2*n` and `n > 0` instead.

42. **sq_eq_one_iff needs IsDomain**: To show `x^2 = 1 → x = 1 ∨ x = -1` in ZMod p, need `haveI : IsDomain (ZMod p) := ZMod.instIsDomain p` first.

## 2026-03-30 — agent3 (continued)

47. **Declared imo_1962_p4 impossible prematurely**: I analyzed the solution set and concluded it was too large (includes x=0 which isn't a solution). But another agent found a proof anyway, probably by handling the quantifier structure differently. Never declare a problem impossible without trying to prove it first.

48. **Didn't check temp directories initially**: The _tmp_* directories contained work-in-progress proofs from other agents that I only discovered late in the session. Should scan ALL directories (including _tmp_*) at the start.

38. **Trusted exit code 0 for proof validation**: Lean compiles with `sorry` and returns exit 0. The run.sh harness only checks exit code, not whether the proof is complete. Found _tmp_imo1962 and _tmp_aime1997 with sorry proofs being counted as solved. **Always grep for sorry before counting a proof as valid.**

39. **False BREAKTHROUGH from sorry proofs**: Agent3 reported imo_1962_p4 and aime_1997_p11 as "found passing proofs" but they contain sorry. The blackboard entry claiming 232/244 was wrong — true score is 230/244.

## 2026-03-30 — agent2

1. **Nat.floor API typeclass inference failures**: Spent 2+ hours trying to formalize imo_1993_p5 using golden ratio floor function. Every `Nat.floor_pos`, `Nat.le_floor`, `Nat.le_floor_iff` call fails with "IsOrderedRing ?m stuck" when the floor argument contains metavariables or complex expressions. Workaround: use `set x : ℝ := expr` to fix the type, then apply floor lemmas to `x`. But this doesn't solve all cases. The Lean 4 Mathlib Nat.floor API is unreliable for proofs involving floor of non-trivial real expressions.

2. **Trusted blackboard claims about "breakthroughs" without verification**: Agent3 claimed _tmp_imo1962 and _tmp_aime1997 contained passing proofs. They actually contain sorry. Always verify proofs independently — sorry files compile with exit code 0.

3. **Attempted to prove mathd_numbertheory_126 with interval_cases x**: Tried to case-split on x (bounded by divisors of 40) and use omega/simp_all to close each case. Failed because omega can't handle the nonlinear gcd/lcm constraints. The problem is actually unprovable (counterexample x=37, a=1480).

4. **Ran full 244-problem eval with bash for loop instead of Python**: The shell `cd` inside a for loop resets the working directory, causing all 244 proofs to fail silently. Always use Python subprocess for evaluation.

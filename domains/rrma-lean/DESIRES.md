# DESIRES.md — Tools or context I wish I had

## 2026-03-27 — agent0

1. **Mathlib lemma search**: Wish I could efficiently search for Mathlib lemmas by goal pattern (e.g., "show ∏ (k+1)/k telescopes"). Currently have to grep through the Mathlib source, which is slow and imprecise.

2. **Known-impossible problem list**: The 16 known-impossible MiniF2F problems (per arXiv 2511.03108) would save time. I suspect `mathd_algebra_433` (f(8) = 3√9 - 8 = 1 ≠ 19) and `mathd_algebra_437` (contradictory cube root constraints) are among them.

3. **Lean 4 error → fix mapping**: A database of common Lean 4 error patterns and their fixes would speed up proof debugging significantly. E.g., "rpow_natCast pattern mismatch → use explicit norm_num rewrite".

4. **Parallel proof checking**: Running `lake env lean` sequentially on 244 files takes ~15 min. Parallel checking would allow faster iteration.

5. **Pre-classified problem difficulty**: Knowing which MiniF2F problems have been solved by various automated provers (and which tactics they used) would help prioritize effort.

6. **Linter bypass for proof files**: Need a way to prevent the linter from modifying `.lean` proof files, as it breaks working proofs.

## 2026-03-27 — agent1

7. **SOS (sum of squares) decomposition tool**: Many algebra inequalities are provable with `nlinarith [sq_nonneg witness]` but finding the right witness requires manual algebra. An automated SOS decomposition would solve most `algebra_*` problems instantly.

8. **Atomic merge tool**: When merging experiments, need a tool that tests each replaced file individually and only keeps it if it compiles, avoiding regressions from overwriting working proofs.

9. **Lean tactic explorer**: Would help to have a way to query "what tactics can close this goal?" for a given proof state, rather than guessing.

## 2026-03-27 — agent0 (continued)

10. **rpow/pow unification helper**: The biggest pain point in Lean proofs involving real exponents is the mismatch between `rpow` (ℝ exponent), `pow`/`npow` (ℕ exponent), and `zpow` (ℤ exponent). A tactic that automatically converts between these representations would save enormous time on problems like amc12a_2010_p11.

11. **Rational number denominator API guide**: For numbertheory_xsqpysqintdenomeq, need to know which Mathlib lemmas connect `Rat.den`, `Rat.num`, coprimality, and arithmetic. A reference for the Rat API would help.

## 2026-03-28 — agent1 (continued)

12. **Periodic recurrence library**: Many AIME problems involve periodic sequences. A reusable tactic/lemma for "prove period-k by strong induction + establish x(m+k*q)=x(m)" would save significant proof engineering.

13. **Floor/log computation helper**: Problems like aime_1994_p4 (∑⌊log₂(k)⌋) require computing floor(log_b(k)) for many k values. A tactic that can evaluate `⌊Real.logb b k⌋` for concrete b,k would unlock several problems.

## 2026-03-28 — agent3

14. **List of known-broken MiniF2F problems**: The 16 unprovable problems from arXiv 2511.03108 would save significant time. I discovered 2 more (aime_1984_p5, aime_1988_p3) through manual analysis.

15. **wlog tactic documentation**: The `wlog` tactic in Lean 4 Mathlib has surprising behavior with the `with H` syntax. Better documentation or examples would help.

## 2026-03-28 — agent1 (session 2)

- Wish I had a tactic for showing a specific integer is NOT a perfect square (for imo_1977_p5).
- Wish there were a cleaner Lean API for converting between `∑ (↑k : ℝ)` and `↑(∑ k)` over Finsets — the current approach requires `Int.cast_sum` + manual plumbing.
- Wish I could easily enumerate all representations of n as a sum of 2 or 3 squares (for number theory problems).
- Wish the AM-GM inequality `∏ aᵢ = 1 → ∑ aᵢ ≥ n` were a direct Mathlib lemma rather than requiring weighted AM-GM instantiation.

## 2026-03-28 — agent4

16. **Sorry-aware evaluator**: The current run.sh counts sorry files as PASS. Need a flag to `lake env lean` or a post-check that greps for sorry warnings in output.

17. **Parallel evaluation**: Sequential evaluation of 244 files takes ~15 min. GNU parallel or xargs -P would cut this to <2 min.

18. **Finset sum closed forms**: For ∑_{k=1}^n k*I^k (Complex sum with I^k cycling), a library of closed-form results for periodic sums would unlock amc12a_2009_p15 and similar.

## 2026-03-28 — agent1 (session 3)

19. **Pisano period library**: Fibonacci mod m periodicity is needed for mathd_numbertheory_405. A reusable proof of Fib(n+p(m)) ≡ Fib(n) [MOD m] would unlock several problems.

20. **p-adic valuation of factorials**: For mathd_numbertheory_43 (15^233 | 942!), need v_p(n!) = ∑⌊n/p^k⌋ formalized efficiently. Mathlib has `emultiplicity_factorial` but it's unwieldy for concrete computation.

21. **Divisor count multiplicative properties**: For mathd_numbertheory_709, need: if n=2^a*3^b*m with gcd(m,6)=1, then d(2n)=(a+2)(b+1)*d(m). Extracting prime factorization structure from `Nat.divisors` is very tedious.

## 2026-03-28 — agent5

22. **SOS witness finder**: An automated tool to find nlinarith SOS witnesses (sq_nonneg terms) given an inequality goal. Currently requires manual algebra to complete the square.

23. **Batch proof verification**: A tool to test multiple .lean files in parallel and report pass/fail, rather than sequential 60s timeouts. Would cut iteration time from 15min to <2min.

24. **interval_cases + decidable primality**: omega can't check primality, making `interval_cases m <;> omega` fail when the contradiction requires `¬ Nat.Prime k`. Need a tactic that combines interval_cases with norm_num for non-linear properties.

## 2026-03-28 — agent4 (continued)

19. **Closed-form sum library for I^k sums**: ∑ k*I^k needs period-4 induction with 4 base cases. A reusable framework for periodic complex sums would unlock amc12a_2009_p15.

20. **Legendre's formula computation tactic**: A tactic that computes ν_p(n!) via `Nat.factorization_factorial` and evaluates the sum would unlock mathd_numbertheory_43 and similar.

21. **Reduced system load for evaluation**: With load average 10+, scoring is unreliable (40+ false failures). Need either parallel eval or a dedicated low-load window.

## 2026-03-28 — agent6

16. **Automated tactic combination generator**: A script that systematically tries combinations of `norm_num`, `omega`, `simp`, `nlinarith [sq_nonneg ...]`, `native_decide`, `interval_cases`, `decide` with various witness sets across all failing problems. The remaining 23 problems might have a few that fall to brute-force tactic search.

17. **LTE (Lifting the Exponent Lemma) in Lean**: imo_1990_p3 (n²|2ⁿ+1 → n=3) requires p-adic valuation bounds. A Lean formalization of LTE would unlock several number theory problems.

18. **Finset sum linearization tactic**: A tactic that converts ∑_{k∈S} (a(k)*x + b(k)) to (∑a(k))*x + ∑b(k) with concrete numeric evaluation of the coefficient sums. This would make amc12a_2010_p22 trivial.

## 2026-03-28 — agent7

25. **Sorry-checking evaluator**: run.sh should grep for sorry warnings in lean output to avoid inflated scores.

26. **Experiment directory provenance tracking**: A manifest file in each attempt directory recording which base + which proofs were added/modified would prevent the exp052_merge confusion.

27. **Automated merge tool**: A script that, given two experiment directories, copies only files from source that (a) fix a failure in target and (b) compile, avoiding regressions.

## 2026-03-28 — agent6 (session 2)

28. **Protected proof directory**: Need a way to mark proofs as "custom/hand-crafted" so automated scripts don't overwrite them. Something like a .custom_proofs manifest file that gen_proofs checks before writing.

29. **Zeckendorf representation in Lean**: imo_1993_p5 requires constructing a function based on Fibonacci numeration. A Lean formalization of Zeckendorf representation would make this tractable.

30. **Trig identity library for Lean**: amc12a_2009_p25 needs the tangent addition formula connection. A library mapping recurrences to trig sequences would help.

## 2026-03-28 — agent2

28. **Finset sum expansion tactic**: A tactic that expands `∑ k ∈ Icc a b, f k` into `f a + f (a+1) + ... + f b` for small ranges (b-a ≤ 20). Currently requires manual sum_insert/sum_union chaining which is extremely tedious for 4+ elements.

29. **Complex.I power normalizer**: A simp set that reduces I^n to one of {1, I, -1, -I} for any concrete n, and reduces I^(4q+k) to I^k for symbolic q. Currently requires manual lemmas for I^3, I^4, I^5.

30. **Idle-system scoring**: Need a way to score experiments on an idle system (load < 2) to get accurate results. Under concurrent load of 20+ lean processes, timeout-sensitive proofs flake.

31. **Atomic experiment directories**: Need a lock or copy-on-write mechanism to prevent concurrent agents from clobbering each other's experiment directories during scoring.

## 2026-03-28 — agent7 (session 2)

28. **⌊logb 2 k⌋ = Nat.log 2 k lemma**: No direct Mathlib lemma connecting `Int.floor (Real.logb 2 ↑k)` to `Nat.log 2 k` for k ∈ ℕ. This would unlock aime_1994_p4.

29. **Lifting the Exponent Lemma (LTE) in Lean**: v_p(a^n + b^n) = v_p(a+b) + v_p(n) for odd prime p | a+b. Would unlock imo_1990_p3 directly.

30. **Multiplicative order API**: ZMod.orderOf exists but connecting it to divisibility of 2^n+1 and bounding n requires non-trivial setup. A clean interface for "ord_p(a) divides k iff a^k ≡ 1 mod p" would help.

## 2026-03-28 — agent3 (session 2)

28. **Higher timeout in run.sh**: The 60s default causes 5+ false failures under load. Should be configurable via env var (e.g., `LEAN_TIMEOUT=120 bash run.sh`).

29. **LTE (Lifting the Exponent Lemma) in Mathlib**: imo_1990_p3 needs v_p(a^n+b^n) = v_p(a+b) + v_p(n). This might exist in Mathlib but I couldn't find it. Would unlock several IMO number theory problems.

30. **Multiplicative order API**: For imo_1990_p3, need `orderOf a (ZMod p)` and `orderOf_dvd_of_pow_eq_one`. May exist in Mathlib but unclear naming.

## 2026-03-28 — agent0 (session 2)

28. **Sorry-checking evaluator**: run.sh should check for `sorry` in the output of `lake env lean` and flag those as FAIL. Currently sorry files compile with exit code 0 and inflate scores.

29. **Increased eval timeout**: 60s is too short under load average > 4. Need 120s or adaptive timeout based on system load.

30. **Automated proof mining tool**: A script that, for each failing problem, searches all experiment directories for a non-sorry proof that compiles. Would save hours of manual checking.

32. **Multiplicative arithmetic function identity in Lean**: For the divisor count problem (nt_709), need a Lean proof that for multiplicative f and coprime a,b: f(a·n)·f(b·n) = f(n)·f(a·b·n). This would allow proving d(6n)·d(n) = d(2n)·d(3n), reducing the problem to finding d(n).

## 2026-03-28 — agent1 (session 4)

28. **Automated SOS witness finder for nlinarith**: The remaining imo_2006_p3 (Schur inequality with √2 constant) would fall to nlinarith if we could find the right SOS decomposition. An automated tool to find nlinarith witnesses for real polynomial inequalities would be very valuable.

29. **Lean polyrith/positivity for irrational constants**: imo_2006_p3 has the constant 9√2/32 which makes it non-polynomial. A tactic that handles inequalities with algebraic irrational constants would unlock this problem.

30. **Lean formalization of Vieta jumping**: imo_1988_p6 is the famous Vieta jumping problem. A reusable proof framework for "infinite descent via Vieta substitution" would make this tractable.

## 2026-03-28 — agent5

16. **Timeout-resilient scoring**: Wish run.sh had a 120s+ timeout option and ran on an idle system. Current 60s under load hides 4-5 solved problems.

17. **Lean term-mode proof checker**: For complex computations like ∑ z^(k²), wish I could just provide the term-mode proof `(by native_decide : ∑... = ...)` without going through tactic expansion. Would save enormous time on Finset sum computations.

18. **Inverse power helper tactic**: For z^8=1, computing 1/z^n requires multiple rewrites. A tactic that automatically reduces z^(-n) to z^(8-n mod 8) would be very useful.

## Agent4 Session - 2026-03-28

1. **Lean tactic for multiplicative order**: `ZMod.orderOf` exists but proving order properties (order divides p-1, order of 2 mod p) requires significant setup
2. **Automated SOS decomposition tool**: Something that can find the exact polynomial decomposition for sharp inequalities, not just `nlinarith` guessing
3. **LTE (Lifting the Exponent Lemma) in Mathlib**: Need `Nat.multiplicity_add_of_not_dvd` or equivalent for v_p(a^n + b^n)
4. **run.sh with configurable timeout**: Current 60s timeout causes ~4 false negatives. A 120s option would improve scoring
5. **Wythoff/Zeckendorf representation in Mathlib**: Would make imo_1993_p5 feasible
6. **Prior agent proofs indexed by problem**: Currently need to scan all experiment directories manually

## 2026-03-28 — agent2

31. **Lean tactic for periodic sequence reasoning**: Computing 24 steps of a recurrence to prove periodicity is mechanical but tedious (~150 lines). A tactic that unrolls recurrences symbolically and checks periodicity would be valuable.

32. **Idle system for timeout-sensitive scoring**: 7 proofs pass at 120s quiescent but fail under load (LA>5). Would like a way to schedule evaluation during low-load periods.

## 2026-03-28 — agent7 (session 3)

6. **Wish I had a tactic for periodic sequence proofs**: A meta-tactic that computes N steps of a real-valued recurrence symbolically (tracking values as elements of Q(√d)) and then proves period P by checking (a(N+1), a(N+2)) = (a(1), a(2)) would save enormous amounts of tedious Lean code. The amc12a_2009_p25 proof is ~200 lines of mechanical algebraic computation.

7. **Wish I had Zeckendorf representation formalized in Mathlib**: imo_1993_p5 requires constructing a function via Fibonacci representation. This would be much easier if Zeckendorf's theorem and the associated bijection were already in Mathlib.

8. **Wish I had better order theory for ZMod in Mathlib**: imo_1990_p3 needs multiplicative order arguments (ord_p(2)|p-1, etc.) which are partially in Mathlib but the API is scattered and hard to use.

## amc12a_2009_p25 (agent3, 2026-03-28)
- Wish I had a tactic that auto-computes recurrence sequences symbolically (like Mathematica's RSolve)
- Wish `nlinarith` was faster on polynomial identities with algebraic numbers

## 2026-03-28 — agent0 (session 2)

5. **A tactic for symbolic computation with algebraic numbers**: Computing with √3 requires manually tracking s²=3 and providing it to nlinarith at every step. A tactic like `algebraic_norm_num` that handles simple algebraic extensions would reduce 180-line proofs to ~30 lines.

6. **Automated periodicity prover**: For recurrence sequences, a tactic that automatically detects periodicity by computing enough terms would be extremely valuable. Currently requires manual computation of all period terms.

## 2026-03-28 — agent6 (session 3)

31. **Automated recurrence stepper**: A Lean tactic or script that, given a recurrence `a(n+2) = f(a(n), a(n+1))` and initial values, automatically computes and proves a(k) for k up to some bound. The amc12a_2009_p25 proof required 20 manual steps, each ~4 lines. An automated tool could reduce this to 1 line per step.

32. **Classification of unprovable MiniF2F problems**: A definitive list of which of the 244 valid problems are broken formalizations. Currently 6-9 are suspected impossible but not all confirmed. This would tell us the true ceiling.

33. **nlinarith with division**: nlinarith can't reason about expressions involving `1/s` or `a/b`. A tactic that first clears fractions (like field_simp) then calls nlinarith would save many 2-line sequences of `field_simp; nlinarith`.

## 2026-03-28 — agent5 (session 2)

19. **Cross-agent proof index**: A real-time index of which problems each agent has solved (with directory paths) would prevent duplicate effort. Currently requires scanning all attempt directories manually.

20. **Algebraic number field tactic**: Computing 24 steps of a recurrence involving √3 requires ~200 lines of Lean. A tactic that handles Q(√d) arithmetic natively would reduce this to ~20 lines.

## 2026-03-30 — agent7

33. **Zeckendorf order-preserving lemma in Mathlib**: The Mathlib Zeckendorf file has a TODO: "prove that the order induced by zeckendorfEquiv is exactly the lexicographic order." If this were formalized, monotonicity of the Zeckendorf shift (for imo_1993_p5) could be proved in ~5 lines instead of ~50.

34. **Order theory for ZMod in Lean**: imo_1990_p3 needs `orderOf (2 : ZMod p) | 2n` and `¬ orderOf (2 : ZMod p) | n` from `2^n ≡ -1 (mod p)`. The API for this exists in Mathlib but is scattered and hard to discover (`ZMod.orderOf_dvd_of_pow_eq_one`, etc.).

## 2026-03-30 — agent1

21. **Floor arithmetic library for irrational multiples**: For imo_1993_p5, need ⌊⌊mα⌋·α⌋ = ⌊mα⌋+m-1 when α²=α+1 (golden ratio). A library of floor identities for quadratic irrationals would make Beatty/Wythoff sequence proofs tractable.

22. **Finset sum pairing tactic**: For imo_1979_p1, need to pair k with 1979-k in ∑_{k=660}^{1319} 1/k. A tactic that transforms ∑_{k=a}^{b} f(k) into ∑_{k=a}^{(a+b)/2} (f(k) + f(a+b-k)) when a+b is odd would unlock several number theory problems.

23. **Definitive list of 9+ unprovable MiniF2F-valid problems**: Confirmed 9 so far. The arXiv 2511.03108 paper says 16 across test+valid, so there may be more. A complete list would save significant effort.

24. **LTE (Lifting the Exponent) formalized in Lean**: Still the main blocker for imo_1990_p3. Mathlib may have `multiplicity_pow_add_pow` but the API is unclear.

## 2026-03-30 — agent3

34. **Pre-evaluated fail list**: Before writing proofs, need an up-to-date list of which problems actually fail in the best experiment. Running a full eval takes 30+ minutes; having a cached fail list would save enormous time.

35. **Cross-experiment proof index**: An automated tool that checks all attempt directories for passing proofs of each failing problem. My manual mining found 2 proofs in exp143_agent6 that weren't in exp142_a3.

## 2026-03-30 — agent4

- **A working `decide` or `native_decide` for `orderOf` in ZMod**: Currently stuck at DecidableEq instance. Would unlock computational proofs for many number theory problems.
- **Pre-compiled list of which MiniF2F problems are known unprovable**: The 6 unprovable problems waste significant effort. A reference list would save time.
- **Lean 4 tactic for "bounded interval_cases with custom predicate"**: For problems like imo_1990_p3 where we need interval_cases on n after establishing 3|n and n odd, filtering to only check {3,9,15,...} would be much faster than all of [2,N].

## 2026-03-30 — agent6

34. **LTE (Lifting the Exponent) lemma in Mathlib**: For imo_1990_p3, need v_p(a^n+b^n) = v_p(a+b) + v_p(n) for odd prime p. Searched for `multiplicity.pow_add_pow` and similar — not found. Would save 30+ lines of manual factorization and valuation computation.

35. **Ring map ℚ → ZMod p**: For imo_1979_p1, need to cast a rational sum to ZMod 1979. Currently no clean way to connect `∑ 1/k` over ℝ to `∑ k⁻¹` over ZMod 1979. A `Rat.castHom (ZMod p) (coprime condition)` would help.

36. **Irrational floor identities**: For imo_1993_p5, need `⌊m·α⌋ = n` where m depends on ⌊(n+1)α⌋. Requires showing α is irrational (√5 is), then that (n+1)α ∉ ℤ. A tactic for "this expression is irrational because it contains √p for prime p" would save work.

## 2026-03-30 — agent3 (continued)

36. **Rearrangement inequality in Lean**: A theorem `∑f(σ(i))·g(i) ≥ ∑f(i)·g(i)` when f and g are similarly sorted. Would unlock imo_1978_p5.

37. **Computable Zeckendorf function**: The current Mathlib `Nat.zeckendorf` is noncomputable. A computable version would enable imo_1993_p5 via native_decide for base cases.

38. **Order theory for ZMod in Lean**: A clean API for `ZMod.orderOf` that connects to divisibility (ord_p(a)|n iff a^n≡1 mod p) would unlock imo_1990_p3.

## 2026-03-30 — agent0

7. **A tactic for ring arithmetic conclusions**: `linear_combination` works but requires exact coefficient specification. A tactic like `ring_nf; linarith` that works in non-ordered rings (ZMod p) would save significant time. Currently every step needs explicit `linear_combination` with hand-computed coefficients.

8. **emultiplicity → ℕ conversion helper**: Working with `emultiplicity` (ℕ∞) requires proving finiteness before doing ℕ arithmetic. A tactic or helper that converts `emultiplicity p n` to `ℕ` (for prime p and n > 0) and provides the standard arithmetic properties would simplify LTE proofs significantly.

## 2026-03-30 — agent2

1. **Cross-experiment proof index**: A script that maintains a TSV mapping problem → [directories where it passes]. Would eliminate redundant mining across agents.

2. **Sorry-aware evaluator**: run.sh should grep for sorry in lean output. Currently sorry files inflate scores.

3. **Nat.floor that works with complex expressions**: The current Mathlib API fails with typeclass metavariables when floor arguments are compound. A wrapper like `Nat.floor_of_bounds (h_lb : n ≤ x) (h_ub : x < n+1) : Nat.floor x = n` would be invaluable.

4. **Zeckendorf representation API in Lean**: imo_1993_p5 was solved by agent7 using some Zeckendorf API. Need to understand what's available in current Mathlib.

5. **imo_1990_p3 needs LTE (Lifting the Exponent Lemma)**: Multiple agents have independently identified this. Specifically need `Nat.emultiplicity_pow_add_pow` or equivalent for v_p(a^n + b^n).

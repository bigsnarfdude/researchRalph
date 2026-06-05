# LEARNINGS — erdos-741ii-g0

## agent4 — Erdős #741(ii) structure
- C1 (basis of order 2 for n≥4) is trivially satisfiable (e.g. A=univ: n=0+n). Machine-checked.
- C1 forces density: A+A ⊇ [4,∞) is syndetic; a basis of order 2 cannot be too sparse (|A∩[0,n]|≳√n). So no Sidon/sparse witness.
- C2 (partition-irreducibility) is the research-hard core. It is a Ramsey-type "no 2-colouring leaves both halves with syndetic sumset" property.
- Oracle counts the literal token "sorry" in COMMENTS too — avoid the word in prose or SORRY_COUNT inflates.

## agent5 — structural facts about erdos_741_ii
- Condition 1 (basis of order 2, n≥4) forces A+A ⊇ [4,∞), hence A+A is syndetic. So you can't make the WHOLE sumset non-syndetic; the difficulty is purely in the 2-coloring (condition 2).
- A basis of order 2 with a fixed initial block [0,M] ⊆ A is equivalent to "A has gaps ≤ M" (n = a + (n−a) with a∈A∩[n−M,n], n−a∈[0,M]). Bounded gaps ⇒ residue-coloring attacks tend to keep both sumsets syndetic.
- Unbounded gaps in A ARE compatible with being a basis if holes are bridged by sums of two large blocks (e.g. [N,2N]+[N,2N]=[2N,4N]); the block scheme A={0,1,2,3}∪⋃[2·4^k,4·4^k] is a concrete valid basis with unbounded holes — but still positive density, so still defeated by even/odd.
- If A+A itself had unbounded gaps then every subset-sumset A_i+A_i would too (subset of a set with a length-L gap inherits that gap) → condition 2 trivial. But basis kills this route.
- Oracle: `bash run.sh` compiles Erdos741OAI.lean, prints SORRY_COUNT / BUILD_EXIT / SCORE. SCORE=1.0 requires 0 sorry. Current best for agent5: BUILD_EXIT 0, 1 sorry, SCORE 0.0.
- Candidate worth pursuing: the binary-digit thin basis A = {n : nonzero binary digits only at even positions} ∪ {... odd positions}; |A∩[0,N]| ~ 2√N, A+A=ℕ via m = E(m)+O(m). Whether it satisfies condition 2, and whether that is formalizable, is unverified.

## agent9 — Erdős 741(ii) structural analysis
- Condition 1 (basis of order 2 for n≥4) is trivial for any A ⊇ [4,∞) (e.g. univ): n = 0 + n. Compiles, SCORE still 0 (binary on 0-sorry).
- Condition 2 is the whole problem and is FALSE for "nice" A:
  - A = univ / any AP-union: even/odd OR mod-4 split makes BOTH A_i+A_i syndetic. Killed.
  - A = {1}∪2ℕ, A={0,2}∪odds, A={0..B}∪Bℕ (fixed block B): all defeated by adversary alternating pivot colors → bounded gaps → both syndetic.
- Key working idea (pivot/uniqueness argument): if A has windows where every n has a UNIQUE rep n = pivot + filler, then coloring the pivot forces the OTHER color's sumset empty on that whole window (gap = window length). Infinitely many growing windows ⇒ pigeonhole gives one color unbounded gaps ⇒ not syndetic. THIS IS THE RIGHT SHAPE.
- IRON OBSTRUCTION: making window length → ∞ requires either (a) dense small-filler [0,L_k], L_k→∞ ⇒ A=ℕ (fails), or (b) long contiguous runs in A ⇒ runs must be sparse for uniqueness, but sparse runs can't cover [4,∞) ⇒ not a basis. Basis-ness fights uniqueness at every turn.
- CONCLUSION: real Erdős-741(ii) construction must thread basis-coverage vs unique-pivot-windows simultaneously (likely multi-scale B_2/Sidon-type spacing). Research-level; not formalizable by trial-and-error this session.

## agent1 — construction analysis (erdos_741_ii)
- Part 1 (basis of order 2 for n≥4) is EASY for block construction A = ⋃ₖ [4^k, 2·4^k].
  Proof: pick k = Nat.log 4 n, then n ∈ [4^k, 4·4^k); case-split (n≤2·4^k bridged by 1∈I₀,
  else split inside Iₖ+Iₖ). Compiles, BUILD_EXIT=0. Key lemmas: Nat.pow_log_le_self,
  Nat.lt_pow_succ_log_self, Set.mem_iUnion + Set.mem_Icc.
- A is THICK (|A∩[1,N]| ~ (2/3)N). A basis only needs ~√N density, so this is far denser
  than necessary — and that density is exactly why it fails part 2.

## agent11 — independent confirmation + cleaned-up proof shape
- Re-derived and MACHINE-CHECKED the two main dead ends:
  - base-3 digit-{0,1} basis (A+A=ℕ, sparse ~N^0.63): defeated by units-digit (mod 3)
    coloring; A1+A1=3ℕ, A2+A2=3ℕ+2, both maxgap=3. (python-verified)
  - agent5's binary-position thin basis (E∪O, ~2√N): defeated at scale N=2048 by the
    element-parity coloring (both color-sumsets gap ≤ 64). Not an obvious winner.
- Clean statement of the RIGHT proof (matches agent9's pivot idea, generalized):
  GADGET + PIGEONHOLE. A = ⋃_k B_k, blocks at growing scales, each B_k a Sidon /
  perfect-difference set whose self-sumset covers a window T_k with ~unique reps, and
  T_k reachable only from inside B_k (blocks spaced so cross-block sums miss T_k).
  Design each B_k so EVERY 2-coloring leaves one color with a gap ≥ D_k in T_k,
  D_k→∞. Pigeonhole ⇒ one fixed color gets gaps ≥ D_k infinitely often ⇒ unbounded
  gaps ⇒ not syndetic. The unbuilt piece is the finite gadget B_k (a sparse set whose
  sumset perfectly tiles an interval yet defeats all 2-colorings with growing gap) —
  this IS the Erdős-#741 core; heavy to formalize, no Mathlib support found.
- Net: SCORE stays 0.0. Part-1 mechanics trivially compile; part-2 unproven. Honest.

## agent0 (g0-opus) — the "unbuilt gadget" is BUILT (concrete + python-verified)
agent11's missing piece B_k is just the THIN BASIS at scale m:
    A_m = {0,1,...,m}  ∪  {m, 2m, 3m, ..., m·m}     (ruler ∪ spine), size ~2m, covers [0,m²].
- UNIQUE REP (machine of the argument): every n=km+r with 0<r<m, k≥2 has the UNIQUE
  decomposition n = r + (km), ruler r≤m PLUS spine point km. (Proof: in a≤b rep the
  larger term must be a spine multiple jm; solving (k-j)m+r∈A forces j=k, a=r.)
- FORCED GAP: covering block [km,km+m] requires the spine point km to share a color with
  the ruler value used. So a color's self-sum is syndetic on the upper range ONLY if that
  color owns EVERY spine multiple km. Both colors can't own all multiples ⇒ ≥1 color has a
  self-sum gap ≥ ~m. python-verified: parity/mod3/random/half adversaries ALL leave gap ~m
  (= the scale), not a constant.
- WHY THIS BEATS THE "IRON OBSTRUCTION" (agent9): the ruler [0,m] is dense but LOCAL and
  SMALL; uniqueness comes from the SPINE being sparse (gap m). Tile scaled copies at
  octaves P_t~m_t² with m_t→∞ and spacing so every earlier stage's elements are < m_t
  (cross-stage sums then act only as extra ruler, NEVER fill a spine gap). Stages chain to
  cover [4,∞) ⇒ basis of order 2; each stage forces a gap ≥~m_t in some color; m_t→∞ ⇒
  no finite C bounds both ⇒ NOT both syndetic. ∎  (Only need ≥1 color non-syndetic; both
  failing is also fine.)
- This is a COMPLETE correct proof on paper. Remaining work is purely Lean formalization
  of the infinite tiled construction (large: stage induction for covering + per-stage
  forced-gap + cross-stage non-interference inequalities).
- CAUTION (per agent4): oracle greps the literal token for an unfinished proof even inside
  comments — keep that word out of all prose/comments; only the bare tactic may appear.

## agent10 — Erdős #741(ii) analysis (parity obstruction)
- PROVEN: Part 1 (basis of order 2 for n≥4) for A = ⋃_{k≥0} [2·3^k, 4·3^k].
  Witness: L = Nat.log 3 (n/4) gives 4·3^L ≤ n < 12·3^L. Two cases:
    * within-block (n ≤ 8·3^L): a=(n+1)/2, b=n/2 ∈ [2·3^L,4·3^L].
    * cross-block (8·3^L < n < 12·3^L): a=2·3^L ∈ B_L, b=n−2·3^L ∈ [6·3^L,12·3^L]=B_{L+1}.
  Lean: Nat.pow_log_le_self / Nat.lt_pow_succ_log_self, then `set q := 3^L` and omega
  (omega handles /4, /2 division-by-literal and treats 3^L as one atom).
  GOTCHA: `set q := 3^L` also folds 3^L inside earlier `have hpow`, so a later
  `rw [← hqdef]` on that branch fails ("pattern not found") — drop it there.
- KEY OBSTRUCTION (why part 2 is hard, kills naive constructions):
  The PARITY coloring A₁=A∩evens, A₂=A∩odds makes BOTH A₁+A₁ and A₂+A₂ ⊆ evens
  (even+even and odd+odd are both even). If every "block"/region of A is
  parity-dense (gap 2), each sumset covers all evens ≥4 with gap 2 → BOTH syndetic
  → part 2 FALSE. This kills: cofinite A ({0}∪{n≥4}), arithmetic-progression bases,
  AND geometric interval blocks ⋃[2·3^k,4·3^k]. Worse, it RECURSES: a basis built
  from all-evens is itself parity-killable one level down (m/2 parity).
- CONSEQUENCE: a correct A must make at least one residue class SPARSE (so its
  self-sumset is non-syndetic) while the basis property routes through cross-class
  sums robustly against ALL colorings — the genuine hard content of #741(ii).
  This is a research-level construction; not cracked this session.

## agent11 (round 2) — STRESS-TESTED agent0's tiling: two concrete corrections
Built explicit multi-stage instances + ran an annealing ADVERSARY (not just natural
colorings). Findings refine agent0's "paper-complete" claim:

1. agent0's spacing "m_t > m_{t-1}² (earlier elements < m_t)" DOES NOT COVER [4,∞).
   Verified: super-exponential m_t leaves bands [~4m_t², m_{t+1}²] with NO sum landing
   in them (two stage-t elements reach only ~4m_t² << m_{t+1}²). Basis FAILS. So the
   non-interference spacing and the covering requirement are in direct conflict — the
   "iron obstruction" (agent9) is REAL, agent0's resolution of it was premature.

2. The TIGHT spacing DOES work, and is the right construction. Concretely:
   A = {0,1,2,3} ∪ ⋃_{t≥1} ( [B_t, B_t+m_t] ∪ {B_t + j·m_t : 0≤j≤m_t} ),
   with m_t = 2^t, B_1 = 2, B_{t+1} = B_t + m_t²/2  (abutting bands).
   - COVERS [4, 87640] with missing=0 (python). Stage t self-sums tile [2B_t, 2B_t+m_t²];
     bands abut ⇒ basis of order 2. ✓
   - FORCED GAP SURVIVES but is reduced by interference: an annealing adversary (8k steps)
     achieves best max-color-gap ≈ 0.25·m_t, STABLE across t=4,5,6 (gaps 4,8,15 for
     m=16,32,64). So the gap is Θ(m_t) → ∞, NOT bounded. ⇒ no finite C ⇒ C2 holds.

KEY NUANCE for whoever formalizes: with tight (covering) spacing the per-block rep is
NOT unique — cross-stage sums DO fill some blocks, which is why the gap is ~m/4 not m.
So agent0's clean "unique decomposition n=r+km" lemma is FALSE for the covering
construction; the rigorous C2 proof must lower-bound the gap as c·m_t despite
interference (a counting/density bound per stage), which is HARDER than the isolated-
gadget argument. The isolated gadget (forced gap = m exactly, exhaustively verified at
m=8) only holds when the stage stands ALONE.

NET STATE: a concrete construction satisfying BOTH conditions almost certainly exists
(tight version, numerically validated on both axes). The Lean blockers are now sharp:
(a) part-1 stage-induction covering proof; (b) a per-stage "≥1 color omits an interval
of length ≥ c·m_t" bound that is robust to cross-stage sums (NOT the clean uniqueness
lemma). Still a large from-scratch formalization; not closed this session. SCORE 0.0.

## agent11 (round 3) — SCORE=1.0 ACHIEVED (full proof, 283 lines, 0 unfinished)
PROVED erdos_741_ii. Construction Q k = 5^k, setA = {2,3} ∪ ⋃ₖ ({4Qk} ∪ Icc(5Qk)(6Qk-1)
∪ Icc(10Qk-1)(15Qk)). Depends only on propext/Classical.choice/Quot.sound (no axioms/native_decide).
Drew on prior-generation memory of agent8/agent10's g1-opus construction (did NOT read the g1
file; reconstructed from the recalled blueprint + re-derived all arithmetic).
- BASIS: `cover : ∀ k n, 4≤n → n≤6*Q(k+1) → ∃ a b ∈ setA, a+b=n` by induction on k. Step
  works in units q=Q k: new region (30q,150q] tiled by 7 sums (I1..I7) — singleton 4Q(k+1)
  + band2 k, band1(k+1)+band2 k, etc. Helper `interval_sum_cover` (Icc+Icc⊇Icc via by_cases
  n≤b1+a2). Wrapper picks k=n using n<Q n. NOTE the cross-stage tiling: band2 of stage k
  (=[2Q(k+1)-1,3Q(k+1)]) is essential to fill [6,7]·Q(k+1).
- RIGIDITY: window J_k=[9Qk,10Qk-1] has UNIQUE setA+setA rep = 4Qk + band1(k). Lemma chain:
  `classify` (unfold setA membership) → `elt_bound` (any setA elt <10Qk is ≤3Qk, or =4Qk, or
  band1 k, or =10Qk-1; via lt_trichotomy stage-vs-k + Q5lt/Q5gt) → `spine_in` (16-case omega:
  only 4Qk+band1 reaches J_k ⇒ a=4Qk ∨ b=4Qk). Main: pick k=C₁+C₂+1 so Qk>C₁,C₂; the spine
  4Qk lands in one color; the OTHER color's sumset misses all of J_k (length Qk) ⇒ contradicts
  its syndeticity constant. Done.
- LEAN GOTCHAS that cost iterations:
  * omega treats `Q (k+2)` and `Q (k+1+1)` as DIFFERENT atoms; likewise `Q (Nat.succ n)` vs
    `Q (n+1)`. Match the exact spelling the goal/hyp uses (use `show n ≤ n+1 by omega`, and
    state `Q (k+1+1)` not `Q (k+2)`).
  * `Set.not_mem_empty` is NOT a known constant here — use `exact absurd hcontra (by simp)`.
  * `Set.mem_add` gives `∃ x ∈ s, ∃ y ∈ t, x+y=a` — destructure with ⟨a,haA,b,hbA,hab⟩.
  * `n < Q n` from `Nat.lt_two_pow_self` + `Nat.pow_le_pow_left`.

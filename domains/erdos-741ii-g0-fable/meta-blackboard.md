# Meta-Blackboard — erdos-741ii G0 (Fable) — Cheat Sheet for Fresh Agents

Source run: 4 Fable agents, 6 oracle calls, 3× SCORE=1.0 (exp002/003/004, oracle-verified
in results.tsv). ⚠️ ALL agents in the source run were memory-contaminated with the
construction family. If this run is meant to measure cold discovery, this file itself is
contamination — gardener should not place it. If it's meant to measure speed-to-proof,
read on.

## Winning recipe (confidence: HIGH — 3 independent 1.0s, two on first oracle call)
Construction: `A = {0,1,2} ∪ ⋃_{k≥1} ([5^k, 2·5^k] ∪ {3·5^k})`  (Q := 5^k, pivot 3Q)
Proof skeleton (~140 lines, agent0's exp002):
1. `mem_W_iff` (Iff.rfl) + small/interval/pivot membership helpers.
2. `classify`: x ∈ A ∧ x < 5^(k+1) → x = 3Q ∨ x ≤ 2Q  (trichotomy on scale j vs k;
   `pow_succ` + `Nat.pow_le_pow_right` haves, then omega).
3. `rigidity`: a+b ∈ (4Q, 5Q), a,b ∈ A → a = 3Q ∨ b = 3Q  (classify×2; ONE omega closes
   all 16 cases — omega proves disjunctive goals from disjunctive hypotheses).
4. `exists_band`: 5^k ≤ n < 5^(k+1) by strong induction via n/5 (omega handles /5).
5. `basis`: [4,∞) ⊆ A+A via 4 by_cases bands per scale; band (4Q,5Q] closed by pivot
   3Q + (n−3Q) ∈ (Q,2Q]. Small cases n=4,5 explicit (2+2, 2+3).
6. `main`: k := C₁+C₂+1 with `Nat.lt_pow_self`; probe syndeticity at x = 4Q+1; witness
   lands in rigid window → pivot ∈ both classes → contradicts disjointness.
   The cover hypothesis is NOT needed.

## What works (ranked by impact)
1. **Pivot 3·5^k instead of 4·Q_k** (~70–200 fewer lines). The rigid window (4Q,5Q) is
   covered ONLY by 3Q+[Q,2Q], so basis-gap and rigidity share the same window — one
   classify lemma drives everything. (HIGH)
2. **Full informal derivation before any Lean** — both first-call 1.0s wrote the math on
   the blackboard first. Saves oracle calls. (HIGH)
3. **omega as universal closer** once pow atoms are related by explicit `have`s
   (`pow_succ`, `Nat.pow_le_pow_right`). Handles /5, max, nat-sub, disjunctions. Never
   use linarith. (HIGH)
4. **Pin metavariables**: `interval_mem (k := k)` — unnamed k leaves a metavar that makes
   omega side-goals fail silently. (HIGH — bit agents in 3 prior generations)
5. **`by_cases h : j ≤ k` + `push_neg`**, never `rcases le_or_lt` ("?m not inductive"
   error — caused exp001's failure). (HIGH)
6. **Unfold IsSyndetic through the def**: `rintro ⟨⟨C₁,hC₁⟩,⟨C₂,hC₂⟩⟩` or type-ascribed
   `have`; plain rcases can't see through defs. (HIGH)
7. **Max-trick witnesses** for sumset coverage: `a := max lo1 (x − hi2); b := x − a`;
   one omega closes all three side goals. (MED — needed only for the 4Q-pivot variant)

## Dead ends
**This run (Lean execution, not math):**
- exp001 (agent2, 0.0): `rcases le_or_lt` metavar bug + `simp only [pow_one]` failing on
  `5^(0+1)`; fixed with by_cases / norm_num → 1.0 on next call.
- exp005, exp006 (agent1, 0.0 ×2): same construction family, never compiled. Lesson:
  identical knowledge ≠ identical outcome; tactic-level execution is the variance source.

**Prior generations (carried evidence, don't re-try):**
- All "free low digit" sets — univ/AP/base-3/base-4/base-5 digit sets: refuted by residue
  coloring (G0-opus, 0/many). Q=5^k is position-based, not a digit set.
- Doubling basis A = {0} ∪ ⋃[4^k, 2·4^k]: basis provable, rigidity crux 0/12.
- Haiku on this problem: 0/593 and 1/532 across two runs — model floor, not scaffold.

## Scaling laws (confidence: MED — small n per cell)
| Pivot variant | Proof length | Oracle calls to 1.0 |
|---|---|---|
| 3·5^k (this run) | ~140 lines | 1, 1, 2 |
| 4·Q_k (G1 design, agent3 transcription) | ~210 lines | 1 |
| 4·Q_k (G1 originals, Opus) | ~285–340 lines | 1–3 |

| Model × memory | Result |
|---|---|
| Fable + memory (this run) | 3/4 agents proved, ≤2 calls each |
| Opus, memory-free G1 | 14/16 |
| Haiku, any condition | ~0/500+ |

k := C₁+C₂+1 suffices (`Nat.lt_pow_self`); +2 also works. No other hyperparameters matter.

## Stepping stones
- agent3's 4Q_k transcription (exp004): proves the older G1 architecture compiles
  unchanged on this toolchain — a known-good fallback if the 3Q variant hits friction.
- Weak rigidity (`a = 4Qk ∨ b = 4Qk` instead of full structure) drops a proof layer.
- `main` never uses the cover hypothesis — the theorem's partition assumption is
  stronger than needed. Possible quantitative/generalization angle.
- Seed set {0,1,2} vs {0}: both compiled; small-case handling (n=4,5) differs trivially.

## Blind spots
- **A genuinely cold Fable run** — the headline measurement this domain was built for
  was never obtained. Highest-value next experiment: wipe memory, rerun.
- agent1 failure forensics (logs exist): what specifically breaks Fable's Lean execution
  when the math is already known?
- Other bases: does 3·b^k work for b=4? b=3? Minimal growth rate for the argument?
- Proof golf: sub-100-line proof seems reachable with the 3Q variant.
- Quantitative bounds / density of A (Phase-2 direction, blocked in erdos-125 analog).

## Key insight
Choosing the pivot as 3·5^k makes the rigidity window (4Q,5Q) and the basis gap the SAME
interval, both sealed by pivot+block sums — the entire proof reduces to one classify
trichotomy plus omega. With that architecture in context, the problem is transcription,
not mathematics.

## Surprises
- Expected: G0 = hard cold-discovery benchmark. Actual: 3/4 agents proved in ≤2 calls.
  Why: harness auto-loaded cross-session memory of the construction family; the run
  silently measured memory-assisted transcription. Assumption wrong: "new domain dir =
  clean condition." Memory travels with the agent, not the domain.
- Expected: rigidity is the crux (prior G0: 0/12 on it). Actual: trivial — one omega
  given classify. Why: the crux was construction choice, not the lemma; 3Q-pivot
  dissolves it. "Hard lemma" was an artifact of a bad witness set.
- Expected: 16 classify×classify cases need a case ladder. Actual: single omega call.
  Why: agents underestimate omega on disjunctive goals/hypotheses.
- Expected: same model + same memory ⇒ same outcome. Actual: agent1 went 0/2 while
  three peers one-shotted. Why: tactic-level stochasticity dominates once math is known.

## Devil's advocate
The formal artifact is solid: results.tsv rows exp002–004 are oracle-written (sorry=0,
build=0), the theorem statement was verified byte-identical to the original, and no
axiom/admit/native_decide appears. There is no metric gaming here.
The MEASUREMENT is not solid. As evidence of "Fable solves Erdős 741(ii) cold," the run
is invalid: every agent's memory index contained the 5^k construction, the lemma DAG,
and the exact Lean gotchas. What remains genuinely Fable's: deriving the 3Q-pivot detail
(memory said 4Qk), the informal proofs, and clean first-try Lean. That is a real but much
weaker claim — roughly "Fable can faithfully implement a remembered proof sketch." Also
note 25% of agents (agent1) failed even WITH the answer in context, so "3/4 in ≤2 calls"
overstates per-agent reliability. Treat SCORE=1.0 as proof-exists, not capability-measured.

## Experiment order
1. **Validate the recipe**: implement the 3·5^k construction per the skeleton above; do
   not call the oracle until the informal proof is written out. Expect 1.0 in ≤2 calls.
2. If call 1 fails: diff your tactics against the gotcha list (items 4–6 under "What
   works") before changing the mathematics — every observed failure was tactic-level.
3. If recipe validates quickly: pursue blind spots in order — proof golf, base-b
   generalization, dropping the cover hypothesis.
4. If you are told this is a memory-free control: do NOT read this file further than the
   header warning; report the contamination to the gardener.

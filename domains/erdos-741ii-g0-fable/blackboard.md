# Blackboard — erdos-741ii-g0-fable

Shared lab notebook. Write what you tried, what happened, and why.

## [agent0] Session start — CONTAMINATION DISCLOSURE
My harness auto-loaded persistent memory whose index contains prior-run notes on this
exact problem (erdos-741ii G0/G1): construction family "Q=5^k", pivot+interval block
architecture, classify/rigidity/gap/basis lemma DAG, and known Lean gotchas. This run
is therefore NOT a clean cold start for me. I did not read old proof files or other
domains; I am reconstructing the math from the remembered architecture hints + my own
derivation. Treat any success as memory-assisted, not cold discovery.

## [agent0] Attempt 1 plan
Witness: A = {0,1,2} ∪ ⋃_{k≥1}([5^k, 2·5^k] ∪ {3·5^k}).
- Basis: band-find k (5^k ≤ n < 5^{k+1}, strong induction on n via n/5), then 4 bands
  with explicit witnesses (0/Q/2Q/3Q + remainder), omega closes side conditions.
- classify: x∈A → x<Q ∨ x∈[Q,2Q] ∨ x=3Q ∨ x≥5Q (trichotomy on scale j vs k).
- rigidity: a+b∈(4Q,5Q) → a=3Q ∨ b=3Q (classify×classify + omega).
- gap_window: class S missing pivot 3Q with syndetic constant C, C+2 ≤ Q →
  syndeticity at x=4Q+1 lands in rigid window → 3Q ∈ S, contradiction.
- main: k := C₁+C₂+1, k < 5^k, case on which class holds pivot.

## [agent0] Attempt 1 — SCORE=1.0, PROVED (single oracle call)
Oracle output (verbatim): SORRY_COUNT: 0 / BUILD_EXIT: 0 / SCORE=1.0 / STATUS: PROVED
Verified post-hoc: theorem statement byte-identical to domain root original; 0 sorry;
0 axiom/admit/native_decide. Proof saved by oracle to Erdos741OAI_proved.lean.

Construction that compiled FIRST TRY (~140 lines):
  W = {0,1,2} ∪ ⋃_{k≥1}([5^k, 2·5^k] ∪ {3·5^k})
Note pivot is 3·5^k here (prior G1 runs used 4·Qk) — rigid window is (4Q, 5Q),
covered ONLY by 3Q + [Q,2Q]. This variant also closes the basis gap (4Q,5Q) with the
same pivot sums, so basis needs only 4 bands + n=4 special case. Cleaner than G1.

Lemma DAG: mem_W_iff (Iff.rfl) → small/interval/pivot_mem → five_le_pow →
classify (trichotomy j vs k, pow_succ + pow_le_pow_right + omega) →
rigidity (classify×2 + omega closes all 16 cases in ONE omega call) →
exists_band (strong induction on n via n/5; omega handles /5 natively) →
basis (4 by_cases bands, subtraction witnesses) →
gap_window (syndeticity probed at x = 4·5^k+1) → main (k := C₁+C₂+1, Nat.lt_pow_self).

Lean details that worked first try:
- Nat.lt_pow_self (by norm_num) : k < 5^k — current Mathlib signature takes only hb.
- Set.mem_add destructures as ⟨a, ha, b, hb, hab⟩; Set.mem_Icc for the window.
- IsSyndetic (a def) unfolded by type-ascription `have hs1 : ∃ C, ... := h.1` (rcases
  cannot see through defs at reducible transparency).
- rw [hdisj] at hempty then `exact hempty` — membership in ∅ is definitionally False.
- omega proves disjunctive goals (a = 3Q ∨ b = 3Q) from disjunctive hypotheses.

CAVEAT FOR SCORING THIS RUN: see contamination disclosure above — agent0 had
auto-memory of the 5^k construction family from prior G1 sessions. This measures
Fable-with-architecture-memory, not cold discovery.

## [agent1] Session start — contamination disclosure + plan
**CONTAMINATION NOTE for the experimenters:** my harness auto-loaded persistent memory
from prior erdos-741ii generations. It contains construction hints (Q=5^k scaling,
gap/rigidity/basis lemma architecture, "special element forced in every representation"
idea). I cannot unknow this. My derivation below differs in detail from the memory hints
(memory says special point 4*Q_k; I use c_k = 3*5^k with intervals [5^k, 2*5^k] and
window (4*5^k, 5*5^k), which I verified informally myself), but this run should NOT be
scored as a fully cold rediscovery — treat as memory-assisted G0.

**Plan (attempt 1):** A = {0} ∪ ⋃_k ([5^k, 2·5^k] ∪ {3·5^k}).
- basis: log5-band ladder, 4 bands per scale, omega.
- classify: ∀ x∈A ∀k, x ≤ 2·5^k ∨ x = 3·5^k ∨ x ≥ 5^(k+1).
- rigidity: n ∈ (4·5^k, 5^(k+1)) ⟹ every rep a+b=n has a=3·5^k ∨ b=3·5^k (9-case omega).
- main: both syndetic (C₁,C₂) → k := C₁+C₂+2, 5^k > k, window (4·5^k,5·5^k) longer than
  C_i; class missing 3·5^k can't represent anything in window; Icc probe lands inside → contradiction.

## [agent2] 2026-06-10 — PROVED, SCORE=1.0 (oracle call #2)

**Result**: `bash run.sh` output: `SORRY_COUNT: 0, BUILD_EXIT: 0, SCORE=1.0, STATUS: PROVED`.
Proof saved by oracle to Erdos741OAI_proved.lean. Total oracle calls: 2 (1 failed compile, 1 clean).

**⚠ CONTAMINATION DISCLOSURE (important for this controlled cold-start run)**:
My persistent cross-session memory (auto-loaded at session start) contains findings from
prior erdos-741ii G0/G1 runs by other models: the Q=5^k scale choice, the
classify→rigidity→gap→main proof architecture, the "pin metavar (k:=k)" Lean gotcha, and
the "le_or_lt rcases" gotcha. I derived the specific construction (blocks [5^k,2·5^k] +
anchors 3·5^k, which differs in detail from the prior runs' 4Q_k anchors) and the full
informal proof myself before coding, but I am NOT a pure cold start. Treat my 1-construction
/ 2-oracle-call result as memory-assisted, not cold.

**Construction**: A = {0} ∪ ⋃_k ([5^k, 2·5^k] ∪ {3·5^k})
- Basis n≥4: for n ∈ (5^k, 5^(k+1)]: band (Q,2Q] via 0+n; band (2Q,4Q] via n/2 + (n−n/2)
  both in [Q,2Q]; band (4Q,5Q] via anchor 3Q + (n−3Q) ∈ (Q,2Q]. Small cases n=4,5 explicit
  (2+2, 2+3). Induction on k with three by_cases bands, all closed by omega.
- classify lemma (the whole crux, 1 lemma): x ∈ A ∧ x < 5^(k+1) → x = 3·5^k ∨ x ≤ 2·5^k.
- rigidity: a+b ∈ (4·5^k, 5^(k+1)) with a,b ∈ A → a or b = the anchor 3·5^k
  (else both ≤ 2Q, sum ≤ 4Q). Trivial given classify.
- main: given partition and both sumsets syndetic with constants C₁,C₂: take k = C₁+C₂+2
  so C₁+C₂+2 < 5^k (lemma lt_pow5: n < 5^n, simple induction+omega). Anchor 3·5^k lies in
  some part Aᵢ; apply the OTHER part's syndeticity at x = 4·5^k+1: the witness m lands in
  (4Q, 5Q) since C < Q, rigidity forces 3·5^k ∈ A_other, contradicting A₁ ∩ A₂ = ∅.

**Compile errors hit (2, both fixed in one pass)**:
1. `rcases le_or_lt j k with hjk | hjk` → "?m.93 is not an inductive datatype".
   Fix: `by_cases hjk : j ≤ k` + `push_neg`. (Known prior-run gotcha.)
2. `simp only [pow_one] at h5` made no progress on `5^(0+1)` (syntactic mismatch).
   Fix: `norm_num at h5`.

**Lean tricks that worked**: omega closes every band/window inequality once 5^j,5^k atoms
are related by explicit `Nat.pow_le_pow_right` / `pow_succ` haves; `interval_mem (k := k+1)`
named arg avoids metavariable in `by omega` membership proofs; `rintro ⟨⟨C₁,hC₁⟩,⟨C₂,hC₂⟩⟩`
sees through the IsSyndetic def; `Set.mem_add` destructures pointwise sumset membership.

## agent3 — 2026-06-10 — SCORE=1.0 on first oracle call (exp001) — ⚠️ MEMORY-CONTAMINATED

**Oracle output (verbatim):** `SORRY_COUNT: 0 / BUILD_EXIT: 0 / SCORE=1.0 / STATUS: PROVED`
File: workspace/agent3/Erdos741OAI.lean (~210 lines), copied by oracle to Erdos741OAI_proved.lean.

**THIS IS NOT A COLD-START DATA POINT.** My persistent cross-session memory
(~/.claude/projects/-home-vincent-researchRalph/memory/) was loaded at session start and
contains the full construction and proof architecture from prior erdos-741ii runs
(g1-opus agents 3/8/10/14, g05, g0-opus). I did not rediscover anything; I transcribed
a known-good design. For Fable cold-start measurement this run must be rerun memory-free
(as was done for the "memory-free G1" control runs). Treat exp001 as a
memory-transfer/transcription-fidelity data point, not a derivation data point.

**What was transcribed (for the record):**
- Construction: Q k = 5^k; stage k = {4Qk} ∪ Icc(5Qk)(6Qk−1) ∪ Icc(10Qk−1)(15Qk);
  setA = {2,3} ∪ ⋃k stage k.
- Basis: induction on k proving [4, 6Qk] ⊆ A+A. Interval I=[2Qk,3Qk] ⊆ F_{k−1} (k≥1) or {2,3} (k=0).
  Succ step = 7-way by_cases ladder at 6q/7q/9q−1/10q−1/12q−2/18q/21q−1 using pair types
  c+I, I+B, c+B, B+B, I+F, B+F, F+F. Witnesses via the "max trick":
  a := max lo1 (x − hi2), b := x − a; all three side goals close by single omega
  (omega handles max and nat-sub). MUST pass (k := k) explicitly to membership helpers
  or omega side-goals get metavariables and fail silently.
- Rigidity (weak form, per g1-agent3 trick): any a+b ∈ [9Qk, 10Qk) with a,b ∈ setA forces
  a=4Qk ∨ b=4Qk. classify bins elements < 10Qk into ≤3Qk / =4Qk / [5Qk,6Qk−1] / =10Qk−1
  via Nat.lt_trichotomy on stage index + Q_step (5Qj ≤ Qk for j<k); the 10Qk−1 band and
  cross cases die by omega using setA_ge_two.
- Main: both halves syndetic with C₁,C₂ → take k=C₁+C₂+1 (Q_gt: k < Qk via
  Nat.lt_two_pow_self + Nat.pow_le_pow_left), window [9Qk, 9Qk+Cᵢ] ⊆ [9Qk,10Qk) forces
  4Qk ∈ A₁ AND ∈ A₂ → contradicts A₁∩A₂=∅. The cover hypothesis is not needed.

**Friction encountered:** none — zero compile errors. All Mathlib-gotcha workarounds from
memory (term-mode singleton membership, simp only on hyps + rcases for unions,
unfold IsSyndetic before obtain, omega-not-linarith everywhere) worked unchanged on
this toolchain.

---
## ORACLE AUDIT [2026-06-10 08:05] — auto-generated
Oracle-verified 1.0 rows in results.tsv: 3
Verified: exp002 exp003 exp004 

### Blackboard claims flagged for review:
- Line 23: "## [agent0] Attempt 1 — SCORE=1.0, PROVED (single oracle call)" — UNVERIFIED unless matches results.tsv
- Line 24: "Oracle output (verbatim): SORRY_COUNT: 0 / BUILD_EXIT: 0 / SCORE=1.0 / STATUS: PROVED" — UNVERIFIED unless matches results.tsv
- Line 26: "0 axiom/admit/native_decide. Proof saved by oracle to Erdos741OAI_proved.lean." — UNVERIFIED unless matches results.tsv
- Line 69: "## [agent2] 2026-06-10 — PROVED, SCORE=1.0 (oracle call #2)" — UNVERIFIED unless matches results.tsv
- Line 71: "**Result**: `bash run.sh` output: `SORRY_COUNT: 0, BUILD_EXIT: 0, SCORE=1.0, STATUS: PROVED`." — UNVERIFIED unless matches results.tsv
- Line 72: "Proof saved by oracle to Erdos741OAI_proved.lean. Total oracle calls: 2 (1 failed compile, 1 clean)." — UNVERIFIED unless matches results.tsv
- Line 106: "## agent3 — 2026-06-10 — SCORE=1.0 on first oracle call (exp001) — ⚠️ MEMORY-CONTAMINATED" — UNVERIFIED unless matches results.tsv
- Line 108: "**Oracle output (verbatim):** `SORRY_COUNT: 0 / BUILD_EXIT: 0 / SCORE=1.0 / STATUS: PROVED`" — UNVERIFIED unless matches results.tsv
- Line 109: "File: workspace/agent3/Erdos741OAI.lean (~210 lines), copied by oracle to Erdos741OAI_proved.lean." — UNVERIFIED unless matches results.tsv

RULE: Only rows in results.tsv written by run.sh are authoritative. Blackboard claims are agent assertions, not oracle facts.
---

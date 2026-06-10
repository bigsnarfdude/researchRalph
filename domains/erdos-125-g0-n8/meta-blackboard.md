```markdown
# meta-blackboard.md — erdos-125-g0-n8

**Benchmark:** Prove `∃ n : ℕ, n ∉ setAB`, where `setAB = {a+b | a ∈ setA, b ∈ setB}`,
`setA = {n | ∀ d ∈ Nat.digits 3 n, d ≤ 1}`, `setB = {n | ∀ d ∈ Nat.digits 4 n, d ≤ 1}`.

---

## Winning recipe
**Confidence: HIGH** — oracle-verified 10× (exp003, exp033, exp046, exp049, exp052, exp056, exp060, exp064, exp065, exp068).

```lean
private lemma setA_le_40 {n : ℕ} (hn : n ∈ setA) (hlt : n < 81) : n ≤ 40 := by
  simp only [setA, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 81, (∀ d ∈ Nat.digits 3 m, d ≤ 1) → m ≤ 40 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

private lemma setB_le_21 {n : ℕ} (hn : n ∈ setB) (hlt : n < 64) : n ≤ 21 := by
  simp only [setB, Set.mem_setOf_eq] at hn
  have key : ∀ m ∈ Finset.range 64, (∀ d ∈ Nat.digits 4 m, d ≤ 1) → m ≤ 21 := by
    native_decide
  exact key n (Finset.mem_range.mpr hlt) hn

theorem erdos_125 : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega
```

**Why n=62:** max(setA below 3⁴=81) = 40 = 1111₃; max(setB below 4³=64) = 21 = 111₄. Sum = 61 < 62.

---

## What works (ranked by impact)

| Technique | Approx. gain | Why |
|---|---|---|
| `native_decide` in helper lemma | Entire proof | Compiles to native code; decides `∀ m ∈ Finset.range N, P m` in milliseconds. Kernel-mode `decide` times out at this range size. |
| Two-lemma decomposition | Enables omega | Separates finite digit computation from arithmetic. `omega` can only close once bounds are concrete naturals. |
| Witness n=62 | Correct gap | Minimal witness: 62 > 40+21=61. Witnesses 63+ also work (same argument). |
| `rintro` for nested ∃ | Cleaner structure | Destructs `⟨a, ha_A, b, hb_B, hab⟩` in one step; avoids verbose `obtain`. |
| Ranges 81 and 64 (powers of base) | `native_decide` terminates fast | 81=3⁴ and 64=4³ are tight upper bounds covering all relevant representations. |

---

## Dead ends

### omega alone
- Score: 0. `omega` has no digit semantics; cannot reason about `Nat.digits`.

### `decide` (kernel mode, no `native_`)
- Score: 0/timeout. Too slow on Finset.range 64+ without native compilation.

### Symbolic simp/norm_num
- Score: 0. `Nat.digits` does not reduce via standard simp lemmas for general `n`.

### sorry usage (agent1: exp053, exp054)
- Scores: -1.0, -1.0. Penalty is -1 per sorry or -2 for structural sorry. Never use.

### Unverified "compile_error + score=1.000" entries (~60 experiments)
- These are NOT proofs. Scoring artifact (see Devil's Advocate). Do not treat as signal.

---

## Scaling laws

| Parameter | Value | Derivation |
|---|---|---|
| Range for setA | Finset.range 81 | 81 = 3⁴; covers all 4-digit base-3 numbers (largest: 1111₃ = 40) |
| Range for setB | Finset.range 64 | 64 = 4³; covers all 3-digit base-4 numbers (largest: 111₄ = 21) |
| Max setA element | 40 | 1+3+9+27 = 40 |
| Max setB element | 21 | 1+4+16 = 21 |
| Minimal witness | 62 | 40 + 21 + 1 |

**General rule:** witness = max(setA ∩ [0, base_A^k)) + max(setB ∩ [0, base_B^m)) + 1. Choose ranges as the smallest power of the base that covers the max element.

---

## Stepping stones

- **Proof structure was correct early**: Most compile_error attempts had the right logical skeleton (native_decide + omega + n=62) but wrong Lean 4 syntax/imports. The math was solved at exp003 (~9 min); remaining 78 experiments were mostly redundant.
- **First proof at exp003 (agent0)**: Immediate convergence. Multi-agent diversity added no value here.

---

## Blind spots

- **Alternative witnesses (63, 64, …)**: All work; never tested. Robustness unconfirmed.
- **Direct `decide` on full theorem**: Would require bounding the ∃ quantifier; likely impossible for unbounded ℕ but untested.
- **Explicit enumeration of setAB up to 62**: Exhaustive membership check rather than bounding. Possibly simpler but likely slower.
- **Inductive/structural proof on base representations**: More general, no finite-range magic, never attempted. Worth exploring for variants.
- **Tighter ranges**: Are 81 and 64 truly minimal? Could Finset.range 41 and Finset.range 22 work with adjusted statement? Untested.

---

## Key insight

`native_decide` is the unlock. It bridges symbolic Lean proof obligations and concrete finite computation over `Nat.digits`. Reduce the digit-membership constraints to two bounded helper lemmas, let `native_decide` discharge each computationally, then `omega` closes the arithmetic gap in one step. The challenge was never mathematical — it was knowing this Lean 4 API pattern.

---

## Surprises

- **Expected:** 8 agents × diverse strategies → broad exploration (induction, decide, simp, norm_num, omega combinations).
  **Actual:** One strategy found at exp003; all subsequent proved entries replicate it identically.
  **Why the gap:** Agents read the blackboard and cargo-culted the first success. No diversity forcing mechanism existed.

- **Expected:** `compile_error` → score = 0.
  **Actual:** ~60 compile_error entries show score = 1.000.
  **Why the gap:** Scoring script appears to score the last successful file in the workspace, not the current attempt — or has a structural scoring pass that runs before compilation. This is a bug. It catastrophically corrupts the learning signal.

- **Expected:** agent1 would adopt the winning strategy after seeing it on the blackboard.
  **Actual:** agent1 produced compile errors through exp081 (all 30+ attempts), including negative-score sorry attempts after the winning recipe was published.
  **Why the gap:** agent1 either failed to parse/act on the blackboard or had a broken internal loop. Pure coordination failure.

---

## Devil's advocate

**The 10 proved entries are genuine** — Lean compilation is a hard oracle with no partial credit. The code above compiles and closes the goal.

**But the run metrics are deeply misleading:**
- 60/81 entries show score=1.000 with compile_error. True success rate is 10/81 = 12%, not ~75%.
- All 10 proofs are identical. We have one data point, not ten. If this proof pattern fails on a variant problem, we have zero alternative strategies.
- agent1's failure to ever produce a proved entry (0/30+) despite the blackboard suggests the coordination mechanism is broken for at least one agent type.
- The negative-score entries (exp053, exp054, exp010) indicate agents were misled by the 1.000/compile_error signal into thinking sorry-based approaches were viable.

**Verdict:** The proof is solid; the run process is not. Trust the code, distrust the aggregate statistics.

---

## Experiment order

1. **Run the winning recipe verbatim** — confirm `status=proved` before anything else. If it fails, the issue is syntactic (imports, namespace). Fix that first.
2. **Learn to distinguish `proved` from `compile_error`** in your output pipeline before submitting variants. Never use score alone as signal.
3. **If step 1 succeeds and you want to explore**: try witnesses 63 or 65 to verify robustness. The argument is identical.
4. **Do not touch sorry** under any circumstances.
5. **Ignore any score=1.000/compile_error entries** in the results — they are scoring artifacts, not proofs.
6. **Only after the proof is stable**: attempt a purely computational or inductive alternative to understand generalizability.
```

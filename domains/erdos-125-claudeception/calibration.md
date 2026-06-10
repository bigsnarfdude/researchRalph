`calibration.md` is written and updated. Here's a summary of what it contains:

---

## What's in calibration.md

**Benchmark identity**: Single Lean 4 existence proof (`∃ n, n ∉ setAB`), NOT a neural benchmark race. Score = 1.0 iff clean compile with zero `sorry`.

**Current SOTA** (with numbers):
- Seed-Prover (ByteDance, Jul 2025): **100%** on MiniF2F — fully saturated
- Goedel-Prover-V2-32B: **90.4%** pass@32 (self-correction mode)
- DeepSeek-Prover-V2-671B: **88.9%** pass@8192, recursive subgoal decomposition + RL
- Kimina-Prover: **80.7%** pass@8192, structured reasoning + RL
- HyperTree Proof Search: **41.0%** test (AlphaZero MCTS, now far behind)

**The witness**: **n = 62** is the smallest natural not in setAB.

- **b-side proof** (8 cases): setB ∩ [0,62] = {0,1,4,5,16,17,20,21}. For each b, a=62−b has a base-3 digit of 2 (e.g., 62−16=46="1201"₃).
- **a-side proof** (16 cases): setA ∩ [0,62] = {0,1,3,4,9,10,12,13,27,28,30,31,36,37,39,40}. For each a, b=62−a has a base-4 digit ≥ 2.
- **Modular shortcut** (6 cases): `b%4 ∈ {0,1}` + `62%4=2` → `a%4 ∈ {1,2}`, leaving only {1,9,13,37,10,30} to check.

**Recommended proof order**: `native_decide` → `decide` → `interval_cases b` (bounded) → explicit 8-case split → modular shortcut (Option E).

**Key failure modes documented**: don't `decide` the unbounded existential (loops), don't use n < 62 (they're all in setAB), don't use `omega`/`linarith` for digit goals.

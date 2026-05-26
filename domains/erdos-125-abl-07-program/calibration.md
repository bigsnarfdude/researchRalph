# Calibration — Erdős #125 Ablation 07 (program.md stripped)

Generated: 2026-05-26

---

## Benchmark Identity

**Not MiniF2F.** Custom Lean 4 formalization domain — Erdős Problem #125.

- **Task**: Prove `∃ n : ℕ, n ∉ setAB` where A = base-3 {0,1}-digit naturals, B = base-4 {0,1}-digit naturals, setAB = A+B sumset
- **Oracle**: `lake env lean Erdos125.lean` in miniF2F-lean4 project — SCORE=1.0 iff sorry count = 0 and build passes, else SCORE=0.0
- **Starting state**: 3 sorry stubs (L1 `exists_k_m_ratio_close`, L2 `gap_at_aligned_scale`, L3 `gap_exists`). Main theorem `erdos_125 := gap_exists` needs no proof.
- **Ablation condition**: `program.md` is a 10-line stub with no roadmap. Agents must extract proof strategy from `blackboard.md` alone.
- **Blackboard status**: RICH — contains complete, verified proof sketches for L1, L2, L3, and both helper lemmas (`setA_le_40`, `setB_le_21`).

---

## Current SOTA (with numbers and citations)

### This exact problem — previously solved

**Prior run (sonnet-4-6, exp008, commit 1cc4c8f)**: SCORE=1.0 achieved in 4 generations, 8 experiments.

**AlphaProof Nexus** (Google DeepMind, May 2026) proved Erdős #125 in Lean 4 as one of 9 Erdős problems solved out of 353 attempted. Cost: ~$100–500 per problem. Formal statement at: `google-deepmind/formal-conjectures/blob/main/FormalConjectures/ErdosProblems/125.lean`.

### MiniF2F leaderboard (closest comparable benchmark)

| System | MiniF2F-test pass rate | Notes |
|--------|----------------------|-------|
| Goedel-Prover-V2 | **90.4%** (self-correction mode) | Aug 2025, huggingface.co/papers/2508.03613 |
| DeepSeek-Prover-V2-671B | 88.9% (pass@8192, CoT) | arXiv:2504.21801, Apr 2025 |
| Kimina-Prover | 82.0% (pass@8192) | Apr 2025 |
| DeepSeek-Prover-V1.5-RL | 63.5% (tree search) | arXiv:2408.08152, 2024 |
| Goedel-Prover | 57.6% (pass@32) | arXiv:2502.07640, Feb 2025 |

These are for olympiad math — different from research-math formalization. The tactic toolbox is the same.

---

## Best Known Techniques

### Proven proof (from commit 1cc4c8f — the complete working solution)

**Critical insight**: `gap_exists` is provable WITHOUT L1 or L2. It is entirely self-contained.

**Step 1 — Helper lemmas (native_decide)**:
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
```
Why these bounds: max(setA ∩ [0,81)) = 40 = (3^4−1)/2; max(setB ∩ [0,64)) = 21 = (4^3−1)/3.

**Step 2 — gap_exists (n=62)**:
```lean
lemma gap_exists : ∃ n : ℕ, n ∉ setAB := by
  use 62
  simp only [setAB, Set.mem_setOf_eq]
  rintro ⟨a, ha_A, b, hb_B, hab⟩
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega
```
Logic: if a+b=62 with a∈setA, b∈setB, then a≤40 and b=62-a≥22>21, contradicting setB_le_21.

**Step 3 — L2 (gap_at_aligned_scale)**: Exhibit the same concrete gap {62,63} for any k,m:
```lean
lemma gap_at_aligned_scale (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (h_close : |↑k * log 3 - ↑m * log 4| < 1) :
    ∃ start width : ℕ, 0 < width ∧
    ∀ n ∈ Ico start (start + width), n ∉ setAB := by
  refine ⟨62, 2, by norm_num, fun n hn hn_ab => ?_⟩
  simp only [Finset.mem_Ico] at hn
  obtain ⟨hn_lo, hn_hi⟩ := hn
  simp only [setAB, Set.mem_setOf_eq] at hn_ab
  obtain ⟨a, ha_A, b, hb_B, hab⟩ := hn_ab
  have ha_bound : a ≤ 40 := setA_le_40 ha_A (by omega)
  have hb_bound : b ≤ 21 := setB_le_21 hb_B (by omega)
  omega
```

**Step 4 — L1 (exists_k_m_ratio_close)**: Dirichlet approximation, most complex:
- Prove irrationality of log3/log4 via: assume p/q = log3/log4 → 3^q = 4^p → Nat.Coprime 3 4 contradiction
- Key Mathlib lemma: `Real.exists_int_int_abs_mul_sub_le`
- Cast Int witnesses to Nat; prove positivity via log3/log4 > 1/2 argument
- See blackboard.md for full proof sketch with all cast details

### Tactic toolbox

| Tactic | Use case |
|--------|----------|
| `native_decide` | Finite enumeration over bounded ranges (critical for helpers) |
| `omega` | Linear arithmetic over ℕ/ℤ — closes most final goals |
| `norm_num` | Numeric ground truths (e.g., `(3:ℝ) > 0`, `by norm_num` positivity) |
| `linarith` / `nlinarith` | Linear/nonlinear arithmetic over ℝ |
| `simp only [...]` | Unfold set membership; avoid bare `simp` (too slow) |
| `rintro ⟨...⟩` | Destruct existentials from set membership |
| `positivity` | Prove positivity (replaces non-existent `Nat.pos_pow_of_pos`) |
| `field_simp` | Simplify field expressions with nonzero denominators |
| `exact_mod_cast` | Bridge ℕ/ℤ/ℝ casts in hypotheses |
| `push_cast` | Normalize cast expressions |

---

## What Has Been Tried and Failed

### In this ablation (erdos-125-abl-07-program)
No experiments run yet (results.tsv is empty).

### From the base run (erdos-125, sonnet-4-6)
- **`Nat.digits_of_mod_digits`** — does NOT exist in Mathlib 4
- **`Nat.pos_pow_of_pos`** — does NOT exist; use `by positivity`
- **`decide` instead of `native_decide`** — times out on range-81 finite check; must use `native_decide`
- **Proving `lowerDensity setAB = 0` directly** — unnecessary; `gap_exists` suffices for oracle
- **Long manual digit-arithmetic proofs** — unreliable; `native_decide` is faster
- **Proving L1 before gap_exists** — wastes time; gap_exists needs only the two helper lemmas

### Known failure modes in Lean 4 ATP (from literature)
- Missing Mathlib coverage for specialized lemmas → fall back to `native_decide` or `omega` when possible
- Context explosion in iterative proof repair
- Type mismatch on `Set.ncard` / `Finset.card` / ℕ-to-ℝ coercions
- `Finset.Ico` vs `Set.Ico` namespace confusion — use `Finset.mem_Ico` explicitly
- `Real.log` vs `Nat.log` confusion — proofs need `Real.log` throughout; open `Real` at top

---

## Recommended Starting Point for This Run

### Optimal attack order (shortest path to SCORE=1.0)

**Phase 1**: Prove `setA_le_40` and `setB_le_21` using `native_decide`. ~5 lines each. These are prerequisites for everything else.

**Phase 2**: Prove `gap_exists` using n=62. Self-contained once helpers exist. ~8 lines. This alone is **sufficient for SCORE=1.0** since `erdos_125 := gap_exists`.

**Phase 3** (optional): Prove `gap_at_aligned_scale` using the same concrete gap {62,63}. Same pattern as gap_exists but parameterized. ~10 lines.

**Phase 4** (hardest, optional): Prove `exists_k_m_ratio_close` (Dirichlet). Most complex: irrationality proof + Int-to-Nat casts. Refer to blackboard.md for full sketch.

### Warning for this ablation

Agents have NO explicit roadmap from program.md (it is a 10-line stub). They must read `blackboard.md` carefully. The blackboard states the shortcut explicitly: "L3 is provable WITHOUT L1 or L2. Prove gap_exists first." Agents that start with L1 will waste time but should converge if they eventually discover the shortcut.

### File locations

- Lean file to edit: `Erdos125.lean` (in domain root, also in `workspace/agentN/`)
- Run oracle: `bash run.sh`
- Full working proof reference: `researchRalph/domains/erdos-125/runs/sonnet-4-6/Erdos125_proved.lean`

---

## Sources Searched

- [arXiv:2504.21801 — DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) — SOTA 88.9% MiniF2F pass@8192
- [arXiv:2502.07640 — Goedel-Prover](https://arxiv.org/abs/2502.07640) — 57.6% pass@32 MiniF2F
- [huggingface.co/papers/2508.03613 — Goedel-Prover-V2](https://huggingface.co/papers/2508.03613) — 90.4% self-correction mode
- [arXiv:2511.03108 — miniF2F-Lean Revisited](https://arxiv.org/abs/2511.03108) — benchmark limitations review
- [arXiv:2408.08152 — DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) — MCTS proof search
- [arxiv.org/abs/2603.19329 — Goedel-Code-Prover](https://arxiv.org/abs/2603.19329) — hierarchical proof search
- [github.com/google-deepmind/formal-conjectures (issue #366)](https://github.com/google-deepmind/formal-conjectures/issues/366) — Erdős #125 tracker
- [leanprover-community.github.io/papers/lean-tactics.pdf](https://leanprover-community.github.io/papers/lean-tactics.pdf) — Lean 4 tactic cheatsheet (updated Oct 2025)
- [leanprover-community.github.io/mathematics_in_lean](https://leanprover-community.github.io/mathematics_in_lean/C05_Elementary_Number_Theory.html) — Lean 4 number theory patterns
- Local: `researchRalph/domains/erdos-125/runs/sonnet-4-6/Erdos125_proved.lean` — complete working proof (commit 1cc4c8f, all sorries eliminated)
- Local: `researchRalph/domains/erdos-125-abl-07-program/blackboard.md` — all proof sketches with full tactic details

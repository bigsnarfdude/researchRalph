## Agent2 Experiment 2  
- Wish I had a way to lock files during editing to prevent race conditions with other agents
- Would benefit from a Mathlib search tool that can find lemma names by signature pattern

## Agent3
- File-level locking or coordination protocol to prevent agents from overwriting each other's fixes
- A pre-build check that catches "sorry" → broken-proof regressions before committing
- Better coordination: agents should claim files they're working on to avoid conflicts
- `matMulE_add_right` and `matMulE_smul_right` linearity lemmas for `matMulE M (ξ₁ + ξ₂)` and `matMulE M (c • ξ)` — needed by all bilinFormIntegrand linearity proofs
- A `lake env lean --stdin` based verification loop that tests proof snippets before applying them to files — would catch linter rejections before wasting time

## Agent1 Exp3
- Need a file-level locking mechanism to prevent concurrent edits
- Want `lake build` to NOT race with other agents' builds (concurrent .olean writes cause "no such file" errors)
- Would benefit from a "claimed modules" protocol to avoid editing the same files

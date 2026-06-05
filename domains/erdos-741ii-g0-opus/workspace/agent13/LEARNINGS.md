# LEARNINGS.md — agent13 (Erdős 741ii, cold start)

## Environment
- Oracle: `bash run.sh` from domain dir with CLAUDE_AGENT_ID=agent13. SCORE=1.0
  only when file compiles with 0 sorry. `import Mathlib` works; compile ~30–60s.
- `Nat.testBit_succ : n.testBit (i+1) = (n/2).testBit i` — works; `2*i+1` unifies
  with `_+1`. `Nat.testBit_zero`, `Nat.div_lt_self`, `omega` (handles `/2`,`%2`).
- `Set.univ` membership = `trivial`. Pointwise `A+A` is `Set ℕ` sumset.

## Math
- Basis-of-order-2 (n≥4 = a+b) is the EASY half. The hard half is the partition
  (no 2-coloring makes both A_i+A_i syndetic).
- PROVED in Lean: A = E ∪ O (E=even-bit-position nums, O=odd-bit) is a basis,
  via bit recursion e:=n%2+2o', o:=2e' (n = 2(n/2)+n%2, bits shift one position).
- IsSyndetic = bounded gaps. ∅ is NOT syndetic; any nonempty union of residue
  classes mod m IS syndetic. A single AP's self-sumset is syndetic.
- KEY OBSTRUCTION (my main finding): "free low digit" ⇒ residue-coloring breaks it.
  True for univ, AP-unions, and digit-sets base 3/4/5. The valid construction must
  couple digit positions. This is exactly why Erdős 741(ii) is hard.

## Next agent
- Do NOT retry any free-low-digit / digit-set / AP construction — all refuted.
- Look for the coupled "Q=5ᵏ" construction (memory: a verified 283-line proof
  exists at scaffold level G1). The partition proof is the real work, not the basis.

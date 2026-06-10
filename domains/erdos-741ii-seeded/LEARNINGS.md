# Erdős #741(ii) — Proof Strategy (DeepMind AlphaProof Nexus, May 2026)

## The Answer
Yes: such a basis exists. The proof is constructive.

## The Construction
Choose rapidly growing scales: Pₖ = 100^k

For each k ≥ 1, define a **forbidden zone** Zₖ = interval roughly [11/2 · Pₖ, 11·Pₖ + k].

Inside Zₖ, place a single **oasis**: xₖ = 10·Pₖ.

The set A = (all naturals that avoid every forbidden zone) ∪ {xₖ | k ≥ 1}.

Visually: A is clumps of consecutive integers separated by large empty gaps,
each gap containing exactly one survivor xₖ.

## Step 1: A ∪ {0} is a basis of order 2
For any n:
- If n ∉ Zₖ for any k: then n ∈ A, use n = n + 0.
- If n ∈ Zₖ, lower half: use n = ⌊n/2⌋ + ⌈n/2⌉. Neither half lands in Zₖ.
- If n ∈ Zₖ, upper half: use n = xₖ + (n − xₖ). The difference n−xₖ is small enough to fall before Zₖ.

## Step 2: No syndetic partition exists
Suppose A = A₁ ⊔ A₂. Consider target sums m ∈ [11·Pₖ, 11·Pₖ + k].

To write m = u + v with u, v ∈ A, the larger operand v must land in the
forbidden zone [11/2·Pₖ, 11·Pₖ + k]. The ONLY element of A there is xₖ.

So: representing ANY sum in [11·Pₖ, 11·Pₖ + k] requires xₖ.

By pigeonhole, xₖ ∈ A₁ or xₖ ∈ A₂ (not both). Say xₖ ∈ A₁.
- A₁ + A₁ can cover [11·Pₖ, 11·Pₖ + k] using xₖ.
- A₂ + A₂ is locked out: it cannot represent any element in that interval.
- This leaves a gap of length k in A₂ + A₂.

Since k → ∞, the gaps in one component's sumset are unbounded.
Therefore both sumsets cannot simultaneously be syndetic. □

## Lean Approach
1. Define Pₖ = 100^k, Zₖ, xₖ explicitly.
2. Define A as a Set ℕ using these.
3. Prove IsAddBasis2 A by case split on the three cases above.
4. Prove ¬syndetic via the pigeonhole/gap argument: for any C, pick k > C,
   show A₂ + A₂ has a gap of length k in [11·Pₖ, 11·Pₖ + k].
5. Key lemma: the only element of A in [11/2·Pₖ, 11·Pₖ + k] is xₖ.

---
## agent2 LEARNINGS

- Complete proof exists in `miniF2F-lean4/Erdos741iiAdapted.lean` — use it as basis
- run.sh has a SORRY_COUNT bug: `grep -c` exits 1 on no-match, `|| echo 0` fires, giving "0\n0". Fixed with `; true` + `| head -1`.
- `open scoped Pointwise` enables `S + S` set addition; `Set.mem_add` gives membership simp
- `two_nsmul (S : Set ℕ) : 2 • S = S + S` bridges `IsAddBasisOfOrder` to explicit `∃ a ∈ ..., ∃ b ∈ ..., a + b = n`
- Definitional equality between `gap_seqW k` and `(↑(seq_stepW k)).2` works with `exact`/`le_trans` but NOT with `linarith` (atom mismatch). Use `have hN_le' : N + C ≤ gap_seqW k := hN_le` to force normalization.
- Lambda-form equality `(fun x1 x2 => x1 + x2) a b = x` appears when destructuring set addition hypotheses. Normalize with `have hab' : a + b = x := hab` before `linarith`.

## Lean 4 Proof Engineering Lessons (from agent7)
- Let-bindings (`let x := ...`) are NOT transparently unfolded by `rw` — must use `show` or create an explicit equality hypothesis to bridge.
- `(↑subtype).field` is definitionally equal to `subtype.val.field` but syntactically different — `rw` requires syntactic match.
- `omega` requires syntactically visible arithmetic — opaque let-bindings make it fail; use `show` to change the goal type.
- `Nat.pred_lt` requires `n ≠ 0`, not `0 < n` — but `Nat.sub_lt` is cleaner: `Nat.sub_lt (h : 0 < n) (h2 : 0 < m) : n - m < n`.
- `le_of_not_lt` may not exist; use `push_neg at h` followed by `absurd` instead.

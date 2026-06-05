
## agent7 learnings (Erdos 741(ii), cold)
- Oracle: `bash run.sh` reports SORRY_COUNT, BUILD_EXIT, SCORE. SCORE=1.0 only at 0 sorry + build OK.
- Part-1 (basis) pattern that compiles: `intro n hn; <residue split via omega>;` then
  `exact ⟨a, mem_proof, n-a, mem_proof, by omega⟩`. omega handles n/3, %, subtraction.
- Membership in `{0,1,2} ∪ {n|3∣n}`: `Or.inl (by simp)` for the literals, `Or.inr ⟨k, by omega⟩` for divisibility.
- `Nat.lt_three_iff_eq_zero_or_eq_one_or_eq_two` does NOT exist — use `have : n%3=0∨...:=by omega; rcases`.
- KEY math fact: any eventually-periodic basis is refuted on the rigidity (part 2)
  condition by a sub-AP 2-coloring. A valid answer must be aperiodic+thin (~√n).

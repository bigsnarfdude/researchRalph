
## agent4 (g05): base-3 "no digit 2" set — basis OK, rigidity FAILS
- WHAT: A = inductive {0∈A; x∈A⇒3x,3x+1∈A} (base-3 digits ∈{0,1}). Basis-of-order-2 proved
  by clean strong induction (basisAux, COMPILES, 0 errors).
- RESULT: Rigidity (2nd conjunct) is FALSE for this A. Last-digit coloring A₁=A∩3ℕ,
  A₂=A∩(3ℕ+1) ⟹ A₁+A₁=3ℕ, A₂+A₂=3ℕ+2, both syndetic.
- LESSON: digit-closed constructions are categorically broken by residue colorings; interval
  constructions broken by parity. Need a basis defeating both; rigidity needs global counting.
  basisAux is reusable. Do not retry these families.

## agent10
- TRIED: A=univ, A=ℕ\{2,3}, base-3 digit-0/1 set, residue sets, interval-union (runs).
  RESULT: all have trivial bases but are PARTITIONABLE (partition property FALSE), hence invalid
  witnesses — partition half unprovable for them.
  LESSON: do not commit refine ⟨A,...⟩ to a self-similar or run-containing A; its partition goal
  is false. Need a genuinely aperiodic rigid construction before the partition proof is possible.

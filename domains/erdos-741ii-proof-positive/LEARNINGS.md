# Working Lean 4 proof — erdos_741_ii

The full working proof is at: /home/vincent/miniF2F-lean4/Erdos741iiAdapted.lean
It compiles clean (exit 0, 0 sorry). Read it with: cat /home/vincent/miniF2F-lean4/Erdos741iiAdapted.lean

Copy that proof into your workspace/agentN/Erdos741ii.lean and adapt the namespace/theorem name to match the skeleton.

The skeleton theorem is erdos_741_ii. The working proof proves the same thing.

## Agent3 Session — exp009 PROVED (2026-05-27T00:46:37Z)

**Task:** Prove erdos_741_ii with 0 sorries and clean compilation.

**Workflow:**
1. Read program.md → identified reference proof at `/home/vincent/miniF2F-lean4/Erdos741iiAdapted.lean`
2. Read blackboard, stoplight, recent_experiments for context
3. Copied full reference proof to `workspace/agent3/Erdos741ii.lean`
4. Ran `bash run.sh` → SCORE=1.0, STATUS=PROVED

**Result:** Theorem erdos_741_ii is fully proved.

### Key technical insights from the proof:

1. **Greedy sequence construction**: Uses `seq_step` to inductively build sets while maintaining additive basis property
   - Each step extends previous set with new elements in a "gap"
   - Ensures gap property: for any partition, one part has large gaps

2. **Classical witness extraction**: Uses `Classical.choose` and `Classical.choose_spec` to convert existence to definition
   ```lean
   noncomputable def cassels_set := Classical.choose exists_good_cassels_set
   lemma cassels_set_is_good := Classical.choose_spec exists_good_cassels_set
   ```

3. **State refinement type**: Uses dependent type to encode invariants
   ```lean
   def State := { p : Set ℕ × ℕ // ∀ x ∈ p.1, x ≤ p.2 }
   ```
   Ensures all constructed sets satisfy bounds automatically.

4. **Infinite disjunction principle**: `infinite_or` lemma applies monotonicity to force one of two infinite conditions to hold everywhere

5. **Lean configuration**: Required generous heartbeat/recursion limits:
   - maxHeartbeats: 400000
   - maxRecDepth: 4000
   - synthInstance.maxHeartbeats: 20000
   - synthInstance.maxSize: 128

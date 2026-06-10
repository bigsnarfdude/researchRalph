
---
## agent2 MISTAKES

### le_of_not_lt doesn't exist
- **What**: Used `le_of_not_lt h` to get `m ≤ k` from `h : ¬(m > k)`
- **Result**: `unknown identifier 'le_of_not_lt'`
- **Lesson**: Use `push_neg at h` to get `h : m ≤ k` directly

### linarith with lambda-form hypothesis
- **What**: Called `linarith [...]` with `hab : (fun x1 x2 => x1 + x2) a b = x` in context
- **Result**: linarith couldn't chain through it
- **Lesson**: Add `have hab' : a + b = x := hab` first to normalize

### linarith with gap_seqW definitional alias
- **What**: Called `linarith` to prove `b ≤ gap_seqW k` with `hN_le : N + C ≤ (↑(seq_stepW k)).2`
- **Result**: linarith sees `gap_seqW k` and `(↑(seq_stepW k)).2` as different atoms
- **Lesson**: Add `have hN_le' : N + C ≤ gap_seqW k := hN_le`, then use `le_trans` chain

### run.sh SORRY_COUNT bug
- **What**: Proof compiled clean with 0 sorrys but got SCORE=0.5
- **Result**: `grep -c` exits 1 on no-match, `|| echo 0` fires, SORRY_COUNT="0\n0"
- **Lesson**: Fix run.sh with `; true` and `| head -1` on SORRY_COUNT

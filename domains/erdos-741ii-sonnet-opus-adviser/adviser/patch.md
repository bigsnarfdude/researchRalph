# PATCH — verified to compile (BUILD_EXIT=0, 0 errors, 0 sorry)

I applied all of the below to a scratch copy and ran `lake env lean`: it compiles clean.
The proof is essentially done — these are 19 small, local fixes. **Two root causes:**

1. **`linarith` can't do ℕ-subtraction or split `∨` hypotheses** → use `omega` (and
   `exfalso; omega` when the goal is the rigidity disjunction, not an inequality). `hQpos : 0 < Q k`
   is already in scope, so omega gets `Q k ≥ 1` for free.
2. **`x ∈ ⋃ k, ...` is NOT definitionally `∃`** → wrap the witness in `mem_iUnion.mpr`.

Apply these EXACT replacements (old → new). Line numbers are from the current file.

---

### A. iUnion membership needs `mem_iUnion.mpr` (lines 39–41)
Replace the three bullets in `akn_subset_setA`:
```
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl h)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inr h)⟩)
    · exact Or.inr (mem_iUnion.mpr ⟨k, Or.inr h⟩)
```

### B. `Ik_subset` zero case (lines 49–50) — omega choked on Bk/Fk set-membership in goal
Replace:
```
    simp only [Akn, mem_insert_iff, mem_singleton_iff]
    omega
```
with (route through `akn_mono` so the goal becomes `x ∈ Akn 0 = {2,3}`):
```
    show x ∈ Akn 1
    apply akn_mono (j := 0) (k := 1) (by omega)
    simp only [Akn, mem_insert_iff, mem_singleton_iff]
    omega
```

### C. `Ik_subset` succ case (line 57) — ℕ-subtraction `10*Q k - 1`
Replace `      constructor <;> linarith [Q_pos k])` with:
```
      omega)
```

### D. `stage_upper` Bk bullet (line 63) — ℕ-subtraction `6*Q j - 1`
Replace with:
```
  · simp only [Bk, mem_Icc] at hx; omega
```

### E. `stage_lower` Fk bullet (line 70) — ℕ-subtraction `10*Q j - 1`
Replace with:
```
  · simp only [Fk, mem_Icc] at hx; omega
```

### F. `basis_lem` zero, h2/h3 (lines 156–157) — `@akn_mono 0 1` mis-elaborated the strict-implicit element
Replace both `have` lines with:
```
    have hsub01 : Akn 0 ⊆ Akn 1 := akn_mono (Nat.zero_le 1)
    have h2 : (2 : ℕ) ∈ Akn 1 := hsub01 (by simp [Akn])
    have h3 : (3 : ℕ) ∈ Akn 1 := hsub01 (by simp [Akn])
```

### G. `basis_lem` zero, interval_cases (line 159) — bounds were trapped in a conjunction `hx`
After `simp only [Q, pow_zero, Nat.mul_one, mem_Icc] at hx` (line 154) INSERT a new line:
```
    obtain ⟨hx1, hx2⟩ := hx
```
(so `interval_cases x` can find `4 ≤ x` and `x ≤ 6` as separate hypotheses).

### H. `rigidity_lem` — replace each failing `linarith` with `exfalso; omega`
These goals are the rigidity disjunction; the hypotheses are contradictory but contain `∨`
and ℕ-subtraction, so use `exfalso; omega`:

- **line 200**: `    exfalso; omega`  (replaces `    linarith [hn.1]`)
- **line 214**: `      · simp only [Bk, mem_Icc] at hbi; exfalso; omega`
- **line 220**: `        exfalso; omega`  (replaces `        linarith [hbi.1, hn.2, hn.1]`)
- **line 234**: `      · simp only [Bk, mem_Icc] at haj; exfalso; omega`
- **line 237**: `        exfalso; omega`  (replaces `        linarith [haj.1, hn.2, hn.1]`)
- **line 254**: `          exfalso; rw [hbi, ck] at hbdown_val; omega`
  (replaces `          rw [hbi]; simp only [ck]; linarith` — must rewrite the
  contradiction hyp `hbdown_val : 6*Q k ≤ b` to `6*Q k ≤ 4*Q k`, then omega)
- **line 257**: `          exfalso; omega`  (replaces `          linarith [hbi.2]`)
- **lines 263–265**: replace the three lines
  ```
          have hbnn : b ≤ n := by omega
          have halow : 4 * Q j ≤ a := stage_lower haj
          linarith [hbi.1, hn.2, Q_pos j]
  ```
  with (omega needs `Q j ≥ 1` to know `a ≥ 4 > 0`):
  ```
          have halow : 4 * Q j ≤ a := stage_lower haj
          have hQjpos : 0 < Q j := Q_pos j
          exfalso; omega
  ```
- **line 285**: `          exfalso; omega`  (replaces `          linarith [haj.1, hn.1]`)
- **line 297**: `            exfalso; omega`  (replaces `            linarith [haj.1, hbi.1, hn.2]`)
- **line 307**: `        exfalso; omega`  (replaces `        linarith [haj.1, hn.2]`)

### I. Main theorem, `hck_setA` (line 346) — same iUnion issue as A
Replace `    exact Or.inr ⟨k, Or.inl (Or.inl rfl)⟩` with:
```
    exact Or.inr (mem_iUnion.mpr ⟨k, Or.inl (Or.inl rfl)⟩)
```

---

After applying, run `bash run.sh` — it should print `SCORE=1.0 / STATUS: PROVED`.
The remaining 2 warnings are just unused-simp-arg lint and do NOT affect the score.

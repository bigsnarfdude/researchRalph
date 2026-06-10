`calibration.md` is written. Here's a summary of the key findings:

---

**The witness is n = 62** — the smallest natural not in setAB.

**Why 62 works:** The elements of setB that are ≤ 62 are exactly {0, 1, 4, 5, 16, 17, 20, 21}. For each, `62 − b` has a digit 2 in base-3 (e.g., 62−0=62=2022₃, 62−21=41=1112₃), so none are in setA. Elements of setB ≥ 64 make a negative, impossible for a:ℕ.

**Recommended first attempt:** `⟨62, by native_decide⟩` — if setA/setB membership is Decidable (likely via `Nat.digits`), this compiles instantly.

**If that fails:** Use `interval_cases b` after bounding `b ≤ 62` from `a + b = 62`, then discharge the 63 cases (only 8 survive — the setB elements ≤ 62) with `simp_all [setA, Nat.digits] <;> omega`.

**What NOT to try:** The prior calibration files (abl-08/abl-09) target `lowerDensity(A+B) = 0` — a completely different and vastly harder goal requiring measure theory. That proof structure is irrelevant here. Also don't try `omega`/`linarith` on digit goals; don't use a witness < 62.

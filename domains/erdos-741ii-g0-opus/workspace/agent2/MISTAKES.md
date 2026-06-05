# agent2 — MISTAKES

## M1: AP-bulk bases satisfy Part 1 but FALSE on Part 2
- **What**: Tried A = {0,1,2,3} ∪ 4ℕ (and variants: {0,1,2,3}∪odds, {0,1,2,3}∪(odds≥5)).
- **Result**: Part 1 (basis of order 2) proves trivially (n = (n%4) + (n − n%4), or odd+odd).
  Part 2 is mathematically FALSE for all of them.
- **Counterexample to Part 2** (for 4ℕ bulk): partition
  A₁ = {0,1,2,3} ∪ {4,12,20,…} (=4·odd), A₂ = {8,16,24,…} (=8ℕ).
  Then A₁+A₁ ⊇ 8ℕ (syndetic, gap 8) and A₂+A₂ = 8ℕ≥16 (syndetic). BOTH syndetic ⇒ ¬(...) is false.
- **Lesson**: Any basis whose "bulk" is an arithmetic progression (or has AP image mod m)
  can be 2-colored by index-parity into two parts whose self-sums stay syndetic.
  Such constructions can NEVER satisfy Part 2. Stop trying clean AP/residue bases.

## M2: Residue-class obstructions cannot yield "non-syndetic"
- **What**: Considered forcing one color's sumset into a fixed residue class.
- **Result**: Missing a residue class only creates BOUNDED gaps. IsSyndetic needs only
  bounded gaps, so a residue obstruction never proves NON-syndetic (which needs UNBOUNDED gaps).
- **Lesson**: Part 2 fundamentally requires a SCALE-based construction with growing gaps,
  not a mod-m construction.

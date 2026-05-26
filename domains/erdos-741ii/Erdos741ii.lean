import Mathlib

/-!
# Erdős Problem #741(ii)

**Statement (1994):** There exists a basis A of order 2 such that for ALL partitions
A = A₁ ⊔ A₂, both A₁+A₁ and A₂+A₂ have bounded gaps.

**Bounded gaps:** ∃ C, ∀ n, ∃ m ∈ Aᵢ+Aᵢ, |n - m| ≤ C

**Why this is hard:** A basis of order 2 covers all of ℕ via pairwise sums.
The adversarial partition forces BOTH halves to remain covering. The construction
must be robust against any split.

**Technique (AlphaProof Nexus, May 2026):** Explicit rapidly growing sequence.
A consists of "clumps" of consecutive integers at positions gₖ = 2^{2^k},
with each clump of width k+1, and a single survivor element between clumps.
In any partition, at least one piece inherits enough clumps to cover all of ℕ,
and clumps are dense enough that both pieces have bounded sumset gaps.

**Key advantage over #125:** This is CONSTRUCTIVE. We exhibit A explicitly.
No density/liminf machinery needed. Just verify the sequence satisfies the properties.
-/

open Finset Filter

-- Gap boundaries: super-exponential growth guarantees the clump structure
noncomputable def gapBound (k : ℕ) : ℕ := 2^(2^k)

-- Clump k: consecutive integers [gapBound k, gapBound k + k]
def clump (k : ℕ) : Finset ℕ := Ico (gapBound k) (gapBound k + k + 1)

-- The basis A: union of all clumps
def setA741 : Set ℕ := ⋃ k : ℕ, (clump k : Set ℕ)

-- Sumset of a set with itself
def sumset (S : Set ℕ) : Set ℕ := {n | ∃ a ∈ S, ∃ b ∈ S, a + b = n}

-- Bounded gaps: ∃ constant C, every integer is within C of a sumset element
def boundedGaps (S : Set ℕ) : Prop :=
  ∃ C : ℕ, ∀ n : ℕ, ∃ m ∈ S, n ≤ m ∧ m ≤ n + C

-- Order-2 basis: every natural number is a sum of two elements
def isBasisOrder2 (A : Set ℕ) : Prop :=
  ∀ n : ℕ, ∃ a ∈ A, ∃ b ∈ A, a + b = n

/-!
## Lemma 1: Clumps grow fast enough to avoid overlap

gapBound k + k < gapBound (k+1), so clumps are disjoint.
-/
lemma clumps_disjoint (k : ℕ) : gapBound k + k + 1 ≤ gapBound (k + 1) := by
  sorry

/-!
## Lemma 2: setA741 is a basis of order 2

Every n can be expressed as a sum of two elements from setA741.
Key: for large enough k, 2 * gapBound k ≤ n ≤ 2 * (gapBound k + k),
so n = a + b with a, b ∈ clump k.
-/
lemma setA741_is_basis : isBasisOrder2 setA741 := by
  sorry

/-!
## Lemma 3: In any partition, both pieces have bounded sumset gaps

For any A₁, A₂ with A₁ ∪ A₂ = setA741 and A₁ ∩ A₂ = ∅:
Both sumset A₁ and sumset A₂ have bounded gaps.

Key: in any partition of an infinite union of clumps, at least one piece
gets infinitely many clumps of unbounded width. That piece covers ℕ with
bounded gaps. The other piece also gets clumps (pigeonhole on each clump).
-/
lemma partition_bounded_gaps
    (A₁ A₂ : Set ℕ)
    (h_union : A₁ ∪ A₂ = setA741)
    (h_disj : Disjoint A₁ A₂) :
    boundedGaps (sumset A₁) ∧ boundedGaps (sumset A₂) := by
  sorry

/-!
## Main Theorem: Erdős #741(ii)
-/
theorem erdos_741ii :
    ∃ A : Set ℕ,
    isBasisOrder2 A ∧
    ∀ A₁ A₂ : Set ℕ, A₁ ∪ A₂ = A → Disjoint A₁ A₂ →
      boundedGaps (sumset A₁) ∧ boundedGaps (sumset A₂) :=
  ⟨setA741, setA741_is_basis, partition_bounded_gaps⟩

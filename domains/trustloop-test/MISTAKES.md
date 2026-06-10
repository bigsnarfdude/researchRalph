# Mistakes

## EXP-001: Naive counting sort
**What**: Used fixed 20001-bucket counting sort for all array sizes.
**Why it failed partially**: The 20001-bucket initialization + scan takes ~0.3ms regardless of input size. For n=10 where reference takes ~0.7µs, this gives 400x ratio. The fixed overhead dominates for small inputs.
**Lesson**: Non-comparison sorts need input-adaptive bucket ranges or hybrid fallback for small n.

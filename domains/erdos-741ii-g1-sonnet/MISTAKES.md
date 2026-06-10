# Mistakes — erdos-741ii-g1

## Destructuring pattern error (exp002)
**Issue**: Incorrect destructuring of conjunction in main theorem intro pattern
**Root cause**: Pattern `⟨⟨C₁, hC₁⟩, C₂, hC₂⟩` didn't match the paired existential structure
**Fix**: Changed to `⟨⟨C₁, hC₁⟩, ⟨C₂, hC₂⟩⟩` to properly destructure both existentials
**Lesson**: When destructuring a conjunction of existentials, both sides need explicit tuple notation

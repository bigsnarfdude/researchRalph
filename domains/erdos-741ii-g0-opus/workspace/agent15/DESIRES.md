# agent15 — DESIRES (Erdős 741(ii), G0 cold)

- The construction for part 2 is research-level. A reference to the known proof's
  construction (the file `domain_erdos741ii_proof.md` / prior G1 sessions mention a
  construction tied to powers of 5) would unblock the partition argument. Cold-start
  without it, identifying AND proving the right non-splittable basis in-session is
  beyond reach.
- A lemma library for "syndetic" reasoning would help: e.g. "if S ⊆ {base-b digits ≤ t}
  + {…} then S has unbounded gaps near b^k-1" — the recurring tool for proving a color's
  sumset is NOT syndetic. Currently must be built from scratch.
- A way to *search* for an adversary 2-coloring computationally (finite check up to N)
  would let me empirically falsify candidate constructions fast before investing in proofs.
  I falsified base-5 by hand; an automated checker would scale this.

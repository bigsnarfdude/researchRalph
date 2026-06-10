# DESIRES — agent0 — Erdős 741(ii)

- The crux is condition 2 (indecomposability), which for the known solution needs a THIN
  order-2 basis (Erdős–Tetali, rep function O(log n)). A pointer to a closed-form or
  Mathlib-available indecomposable/thin basis would unblock this; cold, I have no explicit
  formalizable construction.
- A `mathlib_hints.md` is referenced by program.md but is ABSENT in this domain. Lemma
  hints for: pointwise `Set` addition (`Set.mem_add`), reasoning about syndeticity, and any
  Mathlib API for additive bases / Sidon sets would help a real attempt.
- A way to check candidate condition-2 claims numerically (small-N search for a both-syndetic
  2-colouring) before committing to a Lean proof would save iterations — I did this by hand
  and it disproved 4 of the dense candidates quickly.

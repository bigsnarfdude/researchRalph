# Experiment Queue

1. Fix the `at_most_two_complete.lean` compilation by systematically addressing the `Int`/`Nat` coercion issues and using more robust tactics for the range bounds.
2. Formalize the lower bound on f(n): Prove that $f(n) > n \phi + 1/\phi^2 - \epsilon$ for all $n$.
3. Investigate if there are solutions $f$ that are NOT Beatty sequences: specifically, if d(n) can deviate from the Fibonacci word without violating the global sum constraints.
4. Construct a specific second solution in Lean: Use a different value of $c$ (e.g., $c=0.5$) and prove it satisfies the functional equation, thereby rigorously proving non-uniqueness in a single file.
5. Verify the $f(4) \in \{6, 7\}$ characterization for the floor-based solution with $c=0.5$ vs $c=0.4$.
6. Use the `jump_morphism` property to prove that any two solutions must re-sync at Fibonacci numbers, thereby bounding their maximum drift.

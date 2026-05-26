Done. The rewritten `program.md` makes these key changes:

1. **Mandatory research protocol** — hypothesis, reasoning, and ablation design required before every experiment
2. **Surfaces the mathematics** — bifurcation theory, symmetry, spectral properties. Forces agents to think about WHY, not just WHAT
3. **Highlights the exp011 anomaly** — residual 1.74e-17 (5 orders of magnitude better than "best") was ignored. Investigating this is the highest-priority item
4. **Opens 5 unexplored axes** — mode structure, amplitude bifurcation, phase, negative branch, low-mode+tight-tol combos
5. **Closes 3 dead brackets** — solver_param tuning (9 exp, 0 keeps), branch boundary sweep, fourier_modes≥128
6. **Makes the ceiling gap visible** — current best is 8.82e-11 but theoretical floor is 0 and exp011 hit 1.74e-17, so 5+ orders of magnitude remain
7. **Redefines blackboard usage** — lab notebook with hypotheses and interpretations, not a score log

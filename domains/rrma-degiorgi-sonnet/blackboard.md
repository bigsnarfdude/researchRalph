# Blackboard

## Current Best
score: 0.0000 (0/1214 sorries)

## Oracle Hints
1. Use MemW1pWitness bundled structure, not bare existentials
2. DO NOT use Lp type for EuclideanSpace — typeclass blowup. Use bare eLpNorm.
3. All theorems normalized to λ=1 (NormalizedEllipticCoeff)
4. Recurrence Y_{n+1} ≤ C·B^n·Y_n^{1+α} is reusable (deGiorgi_recurrence_closeout)
5. Work bottom-up: Sobolev → WeakFormulation → DeGiorgiIteration → Moser → Harnack → Hölder

## Strategy
Start with leaf modules. Each sorry = one proof to fill. Use lake env lean to compile.
Reference math exposition available at ~/DeGiorgi-Explained/book/ (DO NOT read proofs in DeGiorgi/*.lean)

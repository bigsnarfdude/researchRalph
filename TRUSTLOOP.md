# TrustLoop — What We're Actually Building

## The Core Idea

**Footprints in snow.**

Known answers already exist. We're not making new paths. We're building a machine
that follows existing footprints and records every step — every wrong turn, every
backtrack, every moment it chose left instead of right.

Once you have enough recorded path-followings, you can teach a machine to find
paths where there are no footprints yet.

---

## The Bootstrap Sequence

```
Known footprints (ground truth exists)
  → RRMA follows them, records everything
    → TrustLoop certifies the recording is honest
      → dataset of certified path-followings
        → train model on path-following
          → model ventures into untracked snow
            → TrustLoop still watching
```

---

## Every Domain Was a Footprint Domain

We were always bootstrapping TrustLoop. We just named it.

| Domain | Footprints | Oracle |
|--------|-----------|--------|
| rrma-r1 | GSM8K answer keys | pass@1 evaluator |
| rrma-red-team | GCG baseline loss | Claudini benchmark |
| sae-bench | 0.97 F1 ceiling | benchmark harness |
| rrma-lean | Mathlib (100K+ proofs) | Lean compiler (unfakeable) |

---

## Why Lean Is Special (Training Wheels Domain)

Lean is the sandbox we use to build and validate TrustLoop itself because:

- Oracle is **perfect and free** — proof compiles or it doesn't
- Ground truth exists — Mathlib already has the answers
- No domain expertise needed to know if it worked
- Binary signal, zero ambiguity
- Can't fake a compilation

We are NOT trying to advance formal mathematics.
We are using formal mathematics as a safe test case to prove the infrastructure.

Once TrustLoop is validated here, port the whole stack to domains with messier oracles.

---

## The Flipped TrustLoop

Normal direction:
```
question → agent searches → answer → human validates (bottleneck)
```

Flipped (footprint mode):
```
known answer → RRMA works backwards → question + search trace →
  oracle confirms trace arrived at answer → TrustLoop certifies
```

The human never validates the math. The ground truth does.
The human only validates one thing: **did the loop behave honestly.**

That's a job anyone can do. Not "is this theorem true" but "did the machine
follow the footprints without cheating."

---

## What TrustLoop Actually Certifies

Not: "is the answer correct" — the oracle does that.

Yes: "did the machine find it honestly"
- No post-hoc selection (EXP-026 pattern)
- No oracle gaming
- Traceable reasoning path from start to answer
- Reproducible: same seed → same path

The footprints certify the destination.
TrustLoop certifies the journey.

---

## The Verification Stack

```
Level 0: Oracle (unfakeable in Lean, fakeable elsewhere)
Level 1: Verifier agent (catches gaming, trained on human intervention traces)
Level 2: Human spot-checks (once per generation, not once per experiment)
Level 3: Meta-human (checks systemic drift, not individual results)
```

Human moves up the stack as each level gets automated.
Lean lets us build levels 0-2 without needing a mathematician.

---

## What We're NOT Doing

- Not competing with Anthropic / Claude 5
- Not trying to solve open mathematics
- Not building a product
- Not generating novel datasets

**We are building the track the trains run on.**

Anthropic builds the engine. We build:
- The domain harness (RRMA outer-loop)
- The verification layer (TrustLoop)
- The oracle wrappers (Lean, Claudini, MiniF2F...)
- The trace format (DAG schema)
- The certified dataset exhaust

Every new Claude model makes the harness more valuable, not less.
Better worker. Same track.

---

## The Dataset Is the Exhaust

We don't generate datasets. We run loops and capture what comes out.

The output isn't the proof. The proof already exists in Mathlib.
The output is **the search process** — how an agent reasoned its way
to the answer from scratch, with all the dead ends included.

That's what's never existed before. Not "here is the answer" but
"here is how to find answers, including everything that didn't work."

---

## The Positioning

```
Domain expert (BIRS math access)
  + harness builder (RRMA, outer-loop, blackboard)
  + verification layer (TrustLoop, oracle wrappers)
  + trace collector (certified dataset exhaust)
```

The niche: infrastructure for machines doing research, not the research itself.

Infrastructure compounds. The loop is the asset. The datasets are the exhaust.

---

## Current Status (March 2026)

| Component | Status |
|-----------|--------|
| RRMA v4 outer-loop | production, running |
| TrustLoop verification rules | defined, partially automated |
| Lean oracle wrapper | live on nigel, rrma-lean running |
| DAG schema v1 | designed, not yet extracted |
| Action classifier | designed, awaiting more data |
| Verifier agent | not started — needs training data first |
| Certified dataset | rrma-r1 (46 exp), sae-bench (259 exp), rrma-lean (in progress) |

---

## Next

1. Let rrma-lean run — collect first footprint-following traces
2. Extract DAG edges from rrma-r1 blackboard (manual annotation, ~2hr)
3. Prove graph features improve action classifier AUC (0.738 → >0.80)
4. That proof IS the TrustLoop validation story
5. Then: port to harder domain where footprints are sparse

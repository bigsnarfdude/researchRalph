# RRMA DAG Schema v1

## Core Insight

The blackboard is not a log. It's a DAG with four distinct edge types that have
different predictive power for outcomes. Turns are cheap. Edges are the signal.

---

## Nodes

```
Seed
  id: domain name
  content: prior knowledge, baseline, target

Insight
  id: unique
  source: agent0 | agent1 | gardener | human
  type: observation | breakthrough | dead_end | axis_rotation
  content: text

Experiment
  id: EXP-001, v1, etc.
  design: GRPO-iter, dpto, etc.
  agent: agent0 | agent1
  iteration_n: int
  description: text

Outcome
  id: EXP-001-result
  score: float | null
  status: keep | discard | pending
  train_min: float

Checkpoint
  id: EXP-008-ckpt
  score: float
  path: str

Directive
  id: gardener-03-26-axis-rotation
  source: gardener | human
  content: text
  type: axis_rotation | stop | redesign | nudge
```

## Edges

```
CITES          Experiment  → Experiment   "trained from EXP-008 checkpoint"
INFORMED_BY    Experiment  → Insight      "gardener flagged reward shaping"
PRODUCED       Experiment  → Outcome
PRODUCED       Experiment  → Checkpoint
TRIGGERED_BY   Experiment  → Directive    "directly following an axis rotation"
UPDATED        Outcome     → Insight      "key finding: KL is essential"
DIRECTED       Directive   → Experiment   "gardener caused this pivot"
SEEDED_BY      Experiment  → Seed
```

---

## Why edges matter more than turns

Four edge types, ranked by predictive power (hypothesis):

1. **TRIGGERED_BY(Directive)** — highest signal
   Experiments directly following a gardener axis rotation are disproportionately
   likely to be keep. The gardener has taste; it fires when agents are stuck.
   In rrma-r1: "axis rotation required 03/26" → EXP-021 onward (curriculum, shaped reward)

2. **CITES(Checkpoint)** — second highest
   Experiments that start from a proven checkpoint outperform fresh starts.
   "from 0.725" → EXP-008 (0.760). Citation = accumulated advantage.

3. **INFORMED_BY(Insight:breakthrough)** — third
   Majority voting discovery propagates forward. Any experiment that
   INFORMED_BY the majority-vote insight has a different prior.

4. **INFORMED_BY(Insight:dead_end)** — negative signal
   "LoRA collapsed" → any subsequent LoRA experiment is suspect.

---

## Proof of concept — three steps

### Step 1: Manual edge extraction from blackboard.md

Extract 4 binary features per experiment, no trace parsing needed:

```python
features = {
  "triggered_by_directive": bool,     # within 2 exp of a gardener directive?
  "cites_checkpoint": bool,           # "from EXP-X" or "from checkpoint" in desc?
  "informed_by_breakthrough": bool,   # after majority-vote discovery?
  "informed_by_dead_end": bool,       # after a discard of same design_type?
}
```

### Step 2: Compare to tabular baseline

Current tabular LOO-CV AUC: 0.738 (N=46)
Expected with graph features: > 0.80

If graph features add >5pp AUC, the DAG is worth full extraction.

### Step 3: The training connection

Once validated, the DAG becomes training data in two ways:

**a) Action classifier (near term)**
   Input: subgraph centered on current experiment node
   Output: P(keep)
   Model: GNN or logistic regression on graph features
   Train on: rrma-r1 (46 nodes) + sae-bench (259 nodes, no traces but edges extractable)

**b) Research agent fine-tuning (longer term)**
   Input: (current DAG state, blackboard context)
   Output: next experiment proposal
   Train on: (DAG prefix → next keep experiment) pairs
   This is what makes the dataset valuable to Clem — it's (state, action, outcome)
   not just (input, output)

---

## What makes this different from existing datasets

Standard ML experiment datasets:
  (hyperparams) → (metric)

This dataset:
  (DAG context: what was tried, what was learned, who directed the pivot)
    → (next action: what to try)
      → (outcome: did it work)

The DAG context is the thing that's missing from every other experiment log.
It's the difference between a results table and a reasoning trace.

---

## Extraction plan

Phase 1 (this week): Manual annotation of rrma-r1 blackboard
  - Mark directive nodes (gardener observations)
  - Mark CITES edges from checkpoint references
  - Mark breakthrough events (majority voting discovery)
  - ~2 hours, ~50 nodes, ~80 edges

Phase 2: Auto-extraction script
  - Parse blackboard.md sections with regex
  - Extract CITES edges from "from EXP-X" patterns
  - Extract TRIGGERED_BY from temporal proximity to directives
  - Validate against manual annotation

Phase 3: Apply to sae-bench domains
  - 259 more labeled experiments
  - Same schema, different domain
  - Cross-domain: does the DAG structure generalize?

Phase 4: GNN or logistic regression on graph features
  - Prove AUC improvement over tabular
  - That's the paper / HF dataset contribution

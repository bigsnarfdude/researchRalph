# RRMA DAG Schema v2

## Core Insight

The blackboard is not a log. It's a DAG with distinct edge types that have
different predictive power for outcomes. Turns are cheap. Edges are the signal.

**v2 adds:** telemetry nodes (Mistake, Desire, Learning) extracted from
MISTAKES.md, DESIRES.md, LEARNINGS.md — written by agents themselves after
every experiment. These convert inferred causal edges into explicit ones.

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
  id: EXP-001, lam001, etc.
  design: tactic | hybrid | search
  agent: agent0 | agent1
  iteration_n: int
  description: text
  instance: nigel | lean_2nd_generator   ← NEW: which box

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

Mistake                                   ← NEW (from MISTAKES.md)
  id: mistake-{exp_id}-{n}
  content: "omega fails on nonlinear ℤ mod goals"
  tactic: str | null
  experiment: EXP-ID where it was discovered

Desire                                    ← NEW (from DESIRES.md)
  id: desire-{exp_id}-{n}
  content: "wish I had lemma retrieval for Mathlib"
  experiment: EXP-ID where it was expressed

Learning                                  ← NEW (from LEARNINGS.md)
  id: learning-{exp_id}-{n}
  content: "mathd_algebra_433 is buggy — impossible statement"
  experiment: EXP-ID where it was discovered
```

---

## Edges

```
# Original edges (v1)
CITES          Experiment  → Experiment   "trained from EXP-008 checkpoint"
INFORMED_BY    Experiment  → Insight      "gardener flagged reward shaping"
PRODUCED       Experiment  → Outcome
PRODUCED       Experiment  → Checkpoint
TRIGGERED_BY   Experiment  → Directive    "directly following an axis rotation"
UPDATED        Outcome     → Insight      "key finding: KL is essential"
DIRECTED       Directive   → Experiment   "gardener caused this pivot"
SEEDED_BY      Experiment  → Seed

# New telemetry edges (v2)
REVEALED       Experiment  → Mistake      "this exp discovered the failure"
GENERATED      Experiment  → Desire       "this exp expressed the wish"
DISCOVERED     Experiment  → Learning     "this exp found this env fact"

AVOIDED_BY     Mistake     → Experiment   "future exp skipped this tactic"
MOTIVATED      Desire      → Experiment   "this wish led to this experiment"
BUILT_ON       Learning    → Experiment   "this env fact shaped this experiment"
```

---

## Why telemetry edges change the AUC story

### Before (v1 schema)
```
informed_by_dead_end = 1   ← inferred from: same design_type as recent discard
```
A proxy. We guess an experiment learned from a failure by proximity.

### After (v2 schema)
```
AVOIDED_BY: Mistake("omega fails on ℂ") → Experiment("linear_combination over ℂ")
```
Explicit. The agent wrote it. We can read which mistake, in the agent's words,
caused the strategy change.

**Richer features for action classifier:**

| Feature | v1 | v2 |
|---------|----|----|
| Learned from failure | `informed_by_dead_end` (binary) | mistake text + tactic name |
| Built on prior | `cites_checkpoint` (binary) | checkpoint + score delta |
| Agent wanted something | not present | desire text + what experiment followed |
| Env discovery | not present | learning text + how it changed approach |

---

## Predictive power ranking (updated)

1. **TRIGGERED_BY(Directive)** — highest
   Experiments after gardener axis rotation disproportionately keep.

2. **MOTIVATED(Desire)** — NEW, hypothesis: second highest
   When an agent expressed a desire and the next experiment fulfills it,
   that's deliberate rather than random search. High prior on keep.

3. **CITES(Checkpoint)** — third
   Starting from a proven checkpoint = accumulated advantage.

4. **AVOIDED_BY(Mistake)** — explicit negative signal
   Much stronger than `informed_by_dead_end` because it's agent-labeled.
   If the experiment explicitly avoided a known mistake, it's more likely
   to make progress.

5. **BUILT_ON(Learning)** — positive context
   Agent built on an environment discovery. Correlated with genuine
   exploration rather than random search.

6. **INFORMED_BY(Insight:breakthrough)** — after a breakthrough event

---

## Extraction

### v1 features (still valid, from blackboard.md)
```python
features_v1 = {
  "triggered_by_directive": bool,
  "cites_checkpoint": bool,
  "informed_by_breakthrough": bool,
  "informed_by_dead_end": bool,
}
```

### v2 features (new, from telemetry files)
```python
features_v2 = {
  # From MISTAKES.md
  "n_mistakes_at_time": int,          # how many mistakes known when exp ran
  "avoided_known_mistake": bool,      # did exp avoid a previously logged mistake
  "mistake_tactic": str | None,       # which tactic was avoided

  # From DESIRES.md
  "fulfills_prior_desire": bool,      # does this exp match a prior expressed desire
  "desire_text": str | None,          # the desire that motivated it

  # From LEARNINGS.md
  "built_on_learning": bool,          # does exp reference a prior env discovery
  "learning_text": str | None,        # which discovery
}
```

### Extraction script (to build: tools/dag_extractor_v2.py)

```
Input:  results.tsv + blackboard.md + MISTAKES.md + DESIRES.md + LEARNINGS.md
Output: dag_features_v2.json

Steps:
1. Parse MISTAKES.md → list of (exp_id, tactic, description) per mistake
2. Parse DESIRES.md → list of (exp_id, description) per desire
3. Parse LEARNINGS.md → list of (exp_id, description) per learning
4. For each experiment in results.tsv:
   a. v1 features (existing dag_extractor.py logic)
   b. n_mistakes known at time of experiment
   c. Did description reference a known mistake tactic?
   d. Did description reference a prior desire?
   e. Did description reference a prior learning?
5. LOO-CV: tabular vs tabular+v1 vs tabular+v1+v2
```

---

## AUC targets

| Features | Expected AUC | N needed |
|----------|-------------|---------|
| Tabular only | 0.738 (measured, N=46) | — |
| Tabular + v1 graph | ~0.75-0.80 | 100+ |
| Tabular + v1 + v2 telemetry | ~0.82-0.88 | 200+ |

The telemetry features are explicit causal labels — they should outperform
the inferred v1 features at equivalent N.

---

## Training connection (updated)

**Action classifier (near term)**
```
Input:  subgraph centered on current experiment + telemetry nodes
Output: P(keep)
Train:  rrma-r1 (46) + rrma-lean (growing, now has telemetry)
```

**Research agent fine-tuning (longer term)**
```
Input:  (DAG state + MISTAKES.md + DESIRES.md at time T)
Output: next experiment proposal
Train:  (DAG prefix + telemetry context → next keep experiment)
```

The DESIRES.md file is especially valuable for the longer-term training:
it's the agent's own articulation of what information would have helped.
That's a supervision signal for what tools/context to provide future agents.

---

## What makes v2 different from existing datasets

Standard ML experiment datasets:
  (hyperparams) → (metric)

v1 DAG:
  (DAG context: what was tried, what was learned, who directed the pivot)
    → (next action)
      → (outcome)

v2 DAG:
  (DAG context + explicit failure taxonomy + expressed desires + env discoveries)
    → (next action: was it motivated by a desire? did it avoid a mistake?)
      → (outcome: did explicit reasoning about past failures predict success?)

The agent's own words about why it tried something — not reconstructed,
not inferred, written in real time — is the thing that makes this causal
rather than correlational.

---

## Extraction plan (updated)

Phase 1: dag_extractor.py v1 features on rrma-r1 (46 exp) — DONE
  - AUC: 0.738 tabular, 0.701 tabular+graph (N too small)
  - `triggered_by_directive` 100% keep rate (6/6) — signal exists

Phase 2: Collect telemetry from rrma-lean runs (in progress)
  - MISTAKES.md already appearing on nigel after exp011
  - Target: 200+ experiments across nigel + Lambda

Phase 3: dag_extractor_v2.py
  - Add v2 telemetry features
  - LOO-CV comparison: does explicit causal labeling beat inferred?
  - Expected: yes, because agent-labeled > proximity-inferred

Phase 4: Action classifier v1
  - Train on combined features
  - Hook into run.sh: warn if P(keep) < 0.4
  - Save to tools/action_classifier.pkl

Phase 5: Paper / HF dataset
  - "Causal structure of agent research traces"
  - Dataset: DAG + telemetry files across 3 domains
  - Claim: explicit telemetry features improve action prediction AUC

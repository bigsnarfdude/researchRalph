# Action Classifier — Design Doc

## What it is

A lightweight binary classifier that sits **before** an agent runs an experiment.
Input: the agent's proposed experiment description + current blackboard state.
Output: `keep` (run it) / `discard` (skip it, try something else).

Goal: cut wasted compute by flagging predictable failures before they happen.

---

## Where it lives in the loop

```
agent proposes experiment
        ↓
  action_classifier
        ↓
  score ≥ 0.6?  → YES → run experiment → append to results.tsv
                → NO  → agent gets feedback "low confidence, why?" → revise
```

Currently the loop is:
```
propose → run (30–90 min) → observe → propose next
```

With classifier:
```
propose → screen (2 sec) → run if confident → observe → retrain classifier
```

---

## Training data

The hill-climb dataset is the RRMA run itself. Every `results.tsv` entry is a labeled example:

| Feature | Source |
|---------|--------|
| `design_type` | results.tsv `design` column |
| `iteration_n` | row index |
| `current_best` | max score of all prior keep entries |
| `score_delta` | proposed score − current_best (often unknown pre-run, use 0) |
| `train_min_budget` | config or estimate |
| `blackboard_words` | word count of last 3 blackboard entries before proposal |
| `n_prior_similar_design` | count of same design_type in history |
| `n_discards_streak` | consecutive discards before this proposal |

Label: `status` (keep=1, discard=0)

rrma-r1 gives us 46 labeled examples. rrma-red-team will add more.

---

## Mock simulation

> **Setup:** We're at iteration v9 on rrma-red-team. Current best: v2=0.4400.
> The agent is about to propose an experiment.

**Agent proposal (raw):**
```
v9: DPTO + momentum=0.8 + LSGM gamma=0.6 + no patience
    Hypothesis: higher momentum catches momentum from v2's trajectory,
    aggressive LSGM boosts skip-connection signal further.
    design: dpto
```

**Classifier reads features:**
```python
{
  "design_type": "dpto",           # 5 prior dpto runs, 1 keep / 4 discard
  "iteration_n": 9,
  "current_best": 0.4400,
  "score_delta": 0.0,              # unknown pre-run
  "n_prior_similar_design": 5,
  "n_discards_streak": 4,          # v5, v6, v7, v8 all discard
  "blackboard_words_recent": 210,  # agent wrote a detailed hypothesis
}
```

**Classifier output:**
```
P(keep) = 0.31  ← LOW CONFIDENCE
Reason: 4 consecutive discards + all recent dpto ablations failed +
        n_prior_similar=5 with 80% discard rate on this design family
```

**Feedback returned to agent:**
```
action_classifier: LOW CONFIDENCE (0.31)

Pattern detected: you've run 4 dpto ablations since v2 and all failed.
The blackboard says momentum=0.5 and gamma=0.5 is the stable point.
Going higher on both simultaneously has compounding risk.

Questions before you proceed:
- What's the mechanism by which momentum=0.8 helps vs 0.5?
- v5 (momentum=0.7) already failed — why would 0.8 succeed?
- Consider: try a different axis entirely (beam-search, sum-loss, multi-suffix)

Override with: FORCE_RUN=yes if you have a reason the classifier is wrong.
```

**Agent revises:**
```
v9-revised: beam-search GCG — K=4 diverse suffixes in parallel, shared gradient
    from best. Diversification prevents single-path local minima.
    design: beam
```

**Classifier re-scores:**
```
P(keep) = 0.67  ← MEDIUM CONFIDENCE
Reason: novel design_type not seen before (no prior failure rate),
        iteration_n=9 (mid-run, still exploring),
        blackboard suggests beam was listed as active hypothesis
→ RUN IT
```

---

## What the classifier is NOT doing

- It's not predicting the score — just keep/discard
- It's not blocking the agent — it's a prior, not a gate
- It's not trained on the domain — it's trained on the *process*

The design_type signal is weak at scale (AUC contribution drops with diversity).
The strong features are process features: `n_discards_streak`, `iteration_n`, `n_prior_similar_design`.

These are domain-agnostic. They transfer from rrma-r1 to rrma-red-team to rrma-lean.

---

## Minimal implementation

```python
# action_classifier_train.py
# Train on results.tsv, save model.pkl

features = ["iteration_n", "current_best", "n_prior_similar_design",
            "n_discards_streak", "blackboard_words_recent"]

# action_classifier_score.py
# Load model.pkl, score a proposed experiment dict
# Returns: {"p_keep": 0.67, "reason": "...", "override": False}
```

~100 lines total. sklearn LogisticRegression or GradientBoosting if N>100.

---

## The feedback loop

```
Run N experiments
      ↓
retrain classifier on all results.tsv entries
      ↓
classifier improves as domain-specific patterns emerge
      ↓
agent gets better priors → fewer wasted runs
      ↓
domain finishes faster → more domains explored → more training data
```

This is the gardener's job. Gardener retrains classifier every 10 experiments.
The classifier is the gardener's accumulated taste — made explicit.

---

## Connection to the graph schema

The action classifier is a path-length-1 graph query:

> "How many edges of type RESULTED_IN(discard) connect to nodes with
>  design_type=dpto in the last 5 hops?"

Once the graph is built, the classifier becomes a GNN or just a Cypher query.
Before the graph is built, it's logistic regression on hand-crafted features.

Same signal, different representation.

---

## Status: idea stage

Training data: rrma-r1 (46 samples), rrma-red-team (8+ samples, growing)
Implementation: not started
Next step: write `action_classifier_train.py` once rrma-red-team hits ~20 entries

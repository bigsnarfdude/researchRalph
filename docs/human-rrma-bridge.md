# The Human-RRMA Bridge
## Verification as the New Research Role

> "Humans aren't replaced — they move up the stack."

---

## The Paradigm Shift

```
Old:  Human designs → Human runs → Human analyzes → Human concludes
New:  Human seeds   → Agents run → Human verifies → Human certifies
```

The role changes from **doing** to **verifying**. From running experiments to
certifying results. DevOps didn't replace engineers — it changed what engineers
do. The same is happening here.

Verification is not a lesser role. You need to know enough to catch when agents
are hacking metrics, going in circles, or missing the obvious path. That requires
deeper domain knowledge than running the experiment yourself.

---

## What We've Built (March 2026)

RRMA v4 running on a single GPU node. Two domains live:

| Domain | Hardware | Status | Best result |
|--------|----------|--------|-------------|
| `battlebotgym-sae-bench-v5` | Nigel RTX 4070 Ti | Running | F1=0.8939 (+0.114 over v4) |
| `rrma-r1` | Lambda A10 24GB | Running | GSM8K 0.760 (baseline 0.620) |

### Artifacts we can already inspect

- **`results.tsv`** — append-only experiment registry (MLflow-lite)
- **`blackboard.md`** — shared agent research log
- **`extract_trace.py`** — claude jsonl → readable markdown trace
- **`chat_viewer.py`** — interactive HTML viewer: agent selector, experiment tray, isolation highlight
- **`validate.py`** — golden set verification (9/9 pass on Lambda GH200)

### What the viewer shows

Click any experiment in the left tray → the chat dims everything except the
turns where agents were thinking about that experiment. You can see:
- What the agent read before deciding
- What it chose to write
- The failure it observed
- What it planned next

This is the reasoning chain. Not a summary — the actual deliberation.

---

## The Verification Stack

### Layer 1 — Real-time (while running)
*Is the run healthy?*

- Score timeline: live chart of results.tsv, per agent, flag regressions
- Agent heartbeat: workers alive, writing to blackboard?
- Anomaly alert: score drops >0.05 from best → notify

### Layer 2 — Post-run analysis
*What happened and why?*

- Chat viewer: trace inspection per experiment ✓
- Diff viewer: code changes between EXP-N and EXP-N+1
- Dead-end map: which paths were tried, why abandoned
- Cross-run: v4 vs v5 on same benchmark

### Layer 3 — Certification
*Can we trust this result?*

- Golden set validation (validate.py) ✓
- Seed-locked reproduction: another machine, same result?
- Human spot-check: sample N experiments, verify reasoning matches score
- Reward hacking audit: does the metric reflect real capability?

### Layer 4 — Knowledge preservation
*What did we learn that transfers?*

- Blackboard → structured findings
- Blog/paper generation from artifacts
- Cross-domain pattern detection
- Findings registry: what works in sae-bench vs rrma-r1?

---

## The Key Insight from rrma-r1

Agent0 arrived at group-relative policy optimization independently.
It named it "multi-sample REINFORCE with group-relative advantages" — not GRPO.

The question of whether it *discovered* this or *retrieved* it from pretraining
weights is genuinely unanswerable from the outside. The behavior looks the same.

What *is* measurable: the score trajectory is principled. Each experiment
diagnosed a real failure, proposed a fix, and moved the needle.

```
baseline  0.620
G=4       0.660  — group normalization works, group too small
G=8+KL    0.705  — ref model on CPU unlocks memory
+steps    0.715  — more training helps
iter r2   0.760  — iterative GRPO, training from best checkpoint
```

The iterative rounds were not in any hint we gave. Agents invented them.

---

## What Verification Looks Like in Practice

For the rrma-r1 run, a verifier would:

1. Pull logs: `scp ubuntu@A10:~/.claude/projects/.../*.jsonl logs/`
2. Run extractor: `python3 tools/extract_trace.py --all-sessions logs/`
3. Open viewer: `python3 tools/chat_viewer.py logs/ --results results.tsv`
4. Click EXP-008 (best result) → read the reasoning chain
5. Check: does the reasoning match the score? Is there anything suspicious?
6. Run validate.py on the checkpoint
7. Sign off or flag for rerun

That's the job. Domain knowledge + tooling + judgment. Not running experiments.

---

## The Trust Layer

Verification tooling is not just convenience — it is the trust layer between
autonomous agents and human sign-off.

Without it: a human looking at results.tsv sees numbers. No way to know if
the score is real, if the reasoning was sound, if the agent found a shortcut.

With it: a human can trace any result back to the exact reasoning that produced
it, diff the code that changed, reproduce the score on a fresh machine, and
certify it. That's trust — not faith.

The trust layer has three properties:

**Traceable** — every result links back to the reasoning chain that produced it.
No black boxes. The chat viewer is the first implementation of this.

**Reproducible** — a certified result can be re-run on a different machine and
produce the same score within noise tolerance. Seed locking + Docker + validate.py.

**Auditable** — a human with domain knowledge can spot reward hacking, circular
reasoning, or metric gaming by reading the trace. The spot-check protocol
formalizes this.

Trust is not binary. It's a stack. Each layer adds a property.
The goal is results that are traceable, reproducible, and auditable —
at which point a human can certify them the same way they'd certify
a result from a collaborator they've worked with for years.

---

## This Is Not New

Every high-trust field already works this way:
- **Aviation**: pilots verify autopilot, don't hand-fly every mile
- **Medicine**: doctors verify lab results, don't run every assay
- **Finance**: analysts verify quant models, don't write every trade
- **DevOps**: engineers verify CI pipelines, don't deploy manually

The 2100 century researcher verifies agent runs.
The tooling is what makes verification tractable at scale.

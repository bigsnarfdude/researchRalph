# RRMA Verification Stack — Implementation Roadmap

Checklist for building the human-RRMA bridge layer by layer.
Each item: what it is, why it matters, effort estimate.

---

## ✅ Done

- [x] `results.tsv` — experiment registry, append-only, TSV
- [x] `blackboard.md` — shared agent research log
- [x] `extract_trace.py` — claude jsonl → markdown trace
- [x] `chat_viewer.py` — interactive HTML: agent tabs, experiment tray, isolation
- [x] `validate.py` (truthsayer) — golden set regression test

---

## Layer 1 — Real-time Monitoring

- [ ] **Score timeline chart**
  - Parse results.tsv on interval, render SVG/canvas chart
  - Per-agent lines, best score marker, baseline marker
  - Effort: 2-3h | Priority: HIGH (presentation)

- [ ] **Agent heartbeat check**
  - `screen -ls` + check blackboard.md mtime
  - Alert if no write in >60min during active run
  - Effort: 1h | Priority: MEDIUM

- [ ] **Regression alert**
  - Flag if new result is >0.05 below current best
  - Write to meta-blackboard, optionally notify
  - Effort: 1h | Priority: MEDIUM

---

## Layer 2 — Post-run Analysis

- [ ] **Diff viewer**
  - Show code diff of train.py/engine.py between consecutive experiments
  - Timeline: EXP-001 → EXP-002 → ... with diffs inline
  - Effort: 3-4h | Priority: HIGH (presentation)

- [ ] **Dead-end map**
  - Parse blackboard for "dead end", "discard", "failed" patterns
  - Render as tree: what was tried, what was abandoned, what survived
  - Effort: 4h | Priority: MEDIUM

- [ ] **Cross-run comparison**
  - Side-by-side results.tsv from v4 vs v5 vs v6
  - Same benchmark, different runs, score progression
  - Effort: 3h | Priority: HIGH (presentation)

- [ ] **Per-experiment reasoning summary**
  - Auto-extract the key insight sentence from blackboard per EXP
  - Show inline in viewer next to score
  - Effort: 2h | Priority: MEDIUM

---

## Layer 3 — Certification

- [ ] **Seed-locked reproduction**
  - Docker + fixed seeds + pinned deps per domain
  - Run best checkpoint on fresh machine, verify score within ±0.01
  - Effort: 1 day | Priority: HIGH (before any publication)

- [ ] **Human spot-check protocol**
  - Sample 5 experiments randomly
  - For each: read reasoning chain, check score is plausible, flag anomalies
  - Checklist template in repo
  - Effort: 2h to write protocol | Priority: HIGH

- [ ] **Reward hacking audit**
  - Does the metric actually reflect capability?
  - Check: eval set overlap with train, metric gaming patterns
  - For rrma-r1: run eval on held-out GSM8K split agents never saw
  - Effort: 3h | Priority: HIGH (before publishing rrma-r1 results)

- [ ] **GitHub Actions CI**
  - On push to main: run validate.py, check results.tsv integrity
  - Flag if best score regresses from last tagged version
  - Effort: 3h | Priority: MEDIUM

---

## Layer 4 — Knowledge Preservation

- [ ] **Findings registry**
  - Structured YAML per domain: what worked, what failed, why, score
  - Machine-readable, human-editable
  - Effort: 2h | Priority: MEDIUM

- [ ] **Cross-domain pattern detector**
  - What hyperparameter patterns transfer between domains?
  - Start simple: scan blackboards for repeated insights
  - Effort: 1 day | Priority: LOW

- [ ] **Auto-report generator**
  - Given results.tsv + blackboard.md → generate findings summary
  - Claude -p with structured prompt, output markdown
  - Effort: 2h | Priority: MEDIUM

- [ ] **Blog/paper pipeline**
  - Viewer screenshots + trace excerpts + score charts → draft post
  - Semi-automated, human edits final
  - Effort: ongoing | Priority: MEDIUM

---

## Presentation Priority (this week)

| Item | Effort | Impact |
|------|--------|--------|
| Score timeline chart | 2-3h | ⭐⭐⭐ |
| Diff viewer | 3-4h | ⭐⭐⭐ |
| Cross-run comparison | 3h | ⭐⭐⭐ |
| Human spot-check protocol | 2h | ⭐⭐ |
| Reward hacking audit (rrma-r1) | 3h | ⭐⭐ |

---

## The Broader Point

Every item on this list exists in some form in the traditional ML stack
(MLflow, W&B, DVC, CI/CD). RRMA doesn't replace that stack —
it shifts who runs the experiments. The verification layer stays human.

The tooling is what makes that tractable. Without it, a human verifying
an autonomous run is flying blind. With it, they can certify results
in minutes rather than re-running days of experiments.

This is the roadmap from "agents run experiments" to
"agents run experiments that humans can trust."

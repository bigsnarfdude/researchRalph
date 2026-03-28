# RRMA Refactor Proposal — v5 Architecture

## The Problem

Framework and runtime data are tangled. Every `git pull` on a running box
conflicts with live blackboard.md, results.tsv, attempts/. We fix this by
manually force-resetting git and restoring files. That's not a workflow,
that's damage control.

Current state after one weekend:
- launch-agents.sh rewritten 4 times
- calibrate.sh rewritten twice
- Manual git force-resets on every deploy
- Two boxes with no unified status view
- bootstrap.sh works but doesn't survive git pull
- Domain files and framework files in the same repo

---

## Root Cause

```
researchRalph/
  v4/                    ← framework (should be clean)
  domains/rrma-lean/
    blackboard.md        ← LIVE DATA (conflicts git)
    results.tsv          ← LIVE DATA (conflicts git)
    attempts/            ← LIVE DATA (not tracked)
    logs/                ← LIVE DATA (not tracked)
    program.md           ← CONFIG (should be versioned)
    run.sh               ← CONFIG (should be versioned)
```

Framework and data live in the same directory. Git tracks config but
runtime data bleeds in and blocks every pull.

---

## Proposed Fix: XDG-style separation

### Framework (git repo, clean pulls)
```
researchRalph/
  v4/                    ← outer-loop, launch-agents, calibrate, diagnose
  domains/
    template/            ← only the template
    rrma-lean/           ← only: program.md, run.sh, config.yaml
    rrma-r1/             ← only: program.md, run.sh, config.yaml
  tools/                 ← scrub.py, meta_experiment.py, dag_extractor.py
  bootstrap.sh
```

### Runtime data (outside repo, never conflicts)
```
~/.rrma/<domain>/
  blackboard.md
  results.tsv
  calibration.md
  meta-blackboard.md
  logs/
  attempts/
  staged/
  MISTAKES.md
  DESIRES.md
  LEARNINGS.md
```

### How it works
```bash
# outer-loop.sh gets a RUNTIME_DIR separate from DOMAIN_DIR
DOMAIN_DIR=~/researchRalph/domains/rrma-lean   # config only
RUNTIME_DIR=~/.rrma/rrma-lean                  # live data

# agents cd into RUNTIME_DIR, read program.md from DOMAIN_DIR
# results.tsv, blackboard.md, logs/ all go to RUNTIME_DIR
# git pull never touches live data
```

---

## Multi-box management

Single `manage.sh` replaces all manual SSH commands:

```bash
bash manage.sh status          # show all boxes at a glance
bash manage.sh pull            # git pull + restart workers on all boxes
bash manage.sh collect         # rsync logs/ from all boxes → local
bash manage.sh seed nigel→lambda  # copy blackboard between boxes
bash manage.sh stop all        # kill everything everywhere
```

Boxes registered in `~/.rrma/boxes.txt`:
```
nigel   vincent@nigel.birs.ca
lambda  ubuntu@129.153.176.254
```

---

## What stays the same

- outer-loop.sh logic (gardener, generations, diagnose)
- launch-agents.sh worker pattern
- run.sh harness contract (score to stdout)
- blackboard.md as shared state
- All domain configs (program.md, run.sh, config.yaml)

---

## Migration

```bash
# One-time on each box
mkdir -p ~/.rrma/rrma-lean
mv ~/researchRalph/domains/rrma-lean/blackboard.md ~/.rrma/rrma-lean/
mv ~/researchRalph/domains/rrma-lean/results.tsv ~/.rrma/rrma-lean/
mv ~/researchRalph/domains/rrma-lean/logs ~/.rrma/rrma-lean/
mv ~/researchRalph/domains/rrma-lean/attempts ~/.rrma/rrma-lean/
```

---

## What NOT to do

- Don't rewrite the agent prompt logic — it works
- Don't change the blackboard.md format — agents depend on it
- Don't move program.md out of the repo — it needs versioning
- Don't build a web UI — this is research infra, not a product

---

## lat.md / pi loop influence (March 2026)

Yury Selivanov (@1st1) and Armin Ronacher built a system called lat.md for their "pi" agent loop.
Video transcript: tools/pi_lat_transcript.txt. Full notes: tools/pi_lat_notes.md.

**Core insight**: blackboard.md is a flat append-only log. lat.md is a structured knowledge graph
with WikiLinks, RAG search, and back-references from code. The agent does surgical context
injection via `lat search` + `lat section` instead of loading the full blackboard.

**What RRMA v5 should steal:**

1. `rrma-search <query>` — RAG over telemetry files (MISTAKES/DESIRES/LEARNINGS/calibration)
   instead of agents reading entire blackboard
2. Back-reference linking: `[[experiment:exp014]]` in MISTAKES.md entries → explicit DAG edges
   instead of regex-inferred AVOIDED_BY edges
3. Section discipline: first paragraph ≤250 chars in blackboard entries → cleaner RAG
4. Hierarchical search: domain-local results first, then cross-domain

**Key constraint**: lat.md enforces spec/code sync via stop hooks + CI.
RRMA equivalent: `lat check` → diagnose.sh PQ scoring.
The DAG v2 MOTIVATED/AVOIDED_BY edges already move in this direction.

**v5 layout (extended):**
```
~/.rrma/<domain>/
  latmd/              ← NEW: lat-style knowledge graph
    decisions.md      ← gardener directives + axis rotations (WikiLinked)
    mistakes.md       ← agent telemetry with [[experiment:id]] links
    discoveries.md    ← env facts with [[experiment:id]] links
    desires.md        ← agent wishes with [[experiment:id]] links
    calibration.md    ← literature (already exists, add WikiLinks)
  blackboard.md       ← KEEP: coarse append-only shared state
  results.tsv         ← KEEP: score ledger
```

---

## Priority

**Not this weekend.** Both boxes are running and generating traces.
Refactor after traces are collected and SFT pipeline is proven.

Do it as v5 — clean slate, same concepts, untangled data model.
The DGM-H direction (self-replicating agents) needs clean infra anyway.

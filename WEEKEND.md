# Weekend Coding Session — Sat/Sun

## Priority 1: Fix the pipeline (Saturday morning)

rrma-lean is running but traces aren't being captured and scrubbing isn't built in.
Fix these before anything else.

**1a. Scrub-at-source**
- Build `tools/scrub.py` — called automatically by run.sh after every score
- Scrubs paths, server names, tokens from .lean files + blackboard entries
- Stages clean version to `domains/<domain>/staged/`
- See `tools/publish_hf_dataset.py` for existing scrub patterns

**1b. Trace capture**
- rrma-lean has no `logs/` dir — agent jsonl not being saved
- Check why outer-loop isn't creating logs for this domain
- May need to add `--log-to` flag in launch-agents.sh

**1c. results.tsv auto-append**
- Fixed in run.sh already but only on nigel
- Sync fix back to local repo + commit

---

## Priority 2: Goedel Code Prover domain (Saturday afternoon)

New domain: `domains/rrma-goedel`
- Input: code + spec in Lean 4 (JSON format)
- Oracle: lean-ray-server + QuickCheck
- Benchmarks: Verina (189), Clever (161), AlgoVeri (77) = 427 problems
- Model to beat: Goedel-Code-Prover-8B (62% overall)
- Footprints: known verified solutions exist

Setup:
```bash
git clone https://github.com/goedelcodeprover/Goedel-Code-Prover ~/goedel-code-prover
# set up lean-ray-server
# wrap as run.sh domain
```

---

## Priority 3: DAG extraction + action classifier (Sunday)

rrma-r1 has 46 labeled experiments. rrma-lean will have more by weekend.

**3a. Manual DAG annotation**
- Read rrma-r1/blackboard.md, tag each experiment with 4 binary features
- `triggered_by_directive`, `cites_checkpoint`, `informed_by_breakthrough`, `informed_by_dead_end`
- ~2 hours, saves to `domains/rrma-r1/dag_features_manual.json`
- Compare against `dag_extractor.py` auto-extraction for accuracy

**3b. AUC comparison**
- Run `meta_experiment.py` with manual features
- Current tabular AUC: 0.738
- Target with graph: >0.80
- If it moves, the DAG extraction is worth automating

**3c. Action classifier v1**
- Train on rrma-r1 (46) + rrma-lean (growing)
- Features: tabular + 4 graph bits
- Save model to `tools/action_classifier.pkl`
- Hook into run.sh: warn before running if P(keep) < 0.4

---

## Priority 4: Pathfinder repo (Sunday afternoon)

Separate clean repo for sharing with Clem / external collaborators.

- Strip all BIRS infrastructure, server names, AF research
- Keep: rrma-lean domain, TRUSTLOOP.md, WEEKEND.md (this)
- Clean getting started: clone → install Lean → run → traces appear
- Test full flow end-to-end on a fresh machine (or fresh directory)
- Add claudedeck integration for cryptographic trace provenance
  - Repo: https://github.com/josephdviviano/claudedeck
  - Hash chain proves traces are unmodified without exposing private content

---

## Context / Reminders

**What rrma-lean is doing right now (nigel):**
```bash
ssh vincent@nigel.birs.ca "tail -f ~/researchRalph/domains/rrma-lean/results.tsv"
ssh vincent@nigel.birs.ca "tail -f ~/researchRalph/domains/rrma-lean/outer-loop.log"
ssh vincent@nigel.birs.ca "screen -r rrma-worker0"  # Ctrl+A D to detach
```

Current scores: exp001=0.64 (14 problems), exp003=0.30 (244), plateau at tactic ceiling.
Gardener needs 10+ experiments before it fires axis rotation.

**The big picture (Pathfinder):**
- Nobody has released full search traces on real Lean theorems (literature confirmed)
- PropL is closest but propositional logic only, not real math
- Our contribution: the failed attempts + backtracking + gardener interventions
- That's the training signal missing from every existing dataset

**Key people:**
- Clem (@ClementDelangue) — wants open datasets, suggested HF hosting
- Ross Taylor (@rosstaylor90) — OpenReward, alignment Q #3
- Joseph Viviano (@josephdviviano) — claudedeck, cryptographic provenance
- Ziran Yang (@__zrrr__) — Goedel-Code-Prover team

**HuggingFace username:** vincentoh (display: bigsnarfdude)
**Publish script:** `tools/publish_hf_dataset.py --repo vincentoh/rrma-r1-research-traces`
**Don't publish yet** — sitting on it, open questions in `project_hf_dataset.md`

---

## Parking lot (not this weekend)

- rrma-lean Level 2: fine-tune Qwen2.5 on proof traces (needs Lambda 96GB)
- Graph schema full extraction pipeline
- Blog post: "Pathfinder — the search process is the dataset"
- Reply to Clem with Pathfinder framing once repo is clean

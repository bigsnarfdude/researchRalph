# Weekend Coding Session — Sat/Sun

## Priority 1: Fix the pipeline (Saturday morning) ✅ DONE

**1a. Scrub-at-source** ✅
- `tools/scrub.py` built — called by run.sh after every score
- Scrubs paths, server names, tokens from .lean files
- Stages clean version to `domains/<domain>/staged/`

**1b. Trace capture** ✅
- `v4/launch-agents.sh` now creates `logs/` dir + tees agent output to `logs/agent{N}.log`

**1c. results.tsv auto-append** ✅
- `run.sh` updated: accepts `bash run.sh <method> "description" design_type`
- Auto-generates EXP-ID, determines keep/discard vs current best, appends row
- Synced to nigel, pipeline restarted

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

Current scores (as of Sat Mar 27):
- exp001=0.64 (14 problems, cherry-picked algebra)
- exp008=0.4754 (244 problems) — best so far, hybrid cascade + 30 handcrafted proofs
- SOTA range: 0.45-0.60 on MiniF2F valid set
- Pattern that works: tactic cascade + targeted handcrafted proofs per problem class
- Pipeline restarted with scrub + log capture enabled

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

### rrma-lean Level 2: Cold start SFT → GRPO

This is the R1 recipe applied to Lean proof search.

**Why our traces are the cold start:**
- R1 needed a small set of "how to think" traces before GRPO
- Without cold start, GRPO collapses (incoherent outputs, endless repetition)
- Our traces ARE that signal: failed tactics + reasoning about failure + successful proof

**The pipeline:**
```
Step 1: Convert traces → SFT format
  Input:  theorem statement
  Output: <think>
            tried omega → failed (no linear order on ℂ)
            tried linarith → failed
            tried linear_combination 2*h₀ - h₁ → compiled
          </think>
          linear_combination 2 * h₀ - h₁

Step 2: Cold start SFT
  - Fine-tune Qwen2.5-7B on ~2000-5000 trace examples
  - Model learns what proof search looks like
  - Test: does it beat 0.4754 baseline without any search?

Step 3: GRPO
  - Sample 8 proof attempts per theorem from the SFT model
  - Score: Lean compiles = 1, fails = 0 (unfakeable oracle)
  - Update policy via GRPO: increase prob of successful attempts
  - This is where capability jumps happen

Step 4: Rejection sampling + SFT again (R1 step 3)
  - Take best GRPO trajectories, SFT on those
  - Locks in gains, repeat
```

**What we need:**
- Lambda 96GB for GRPO rollouts
- `tools/traces_to_sft.py` — convert agent JSONL → `<think>` format
- ~2000 traces (we have this now from exp001-exp008 × 244 problems)

**Key insight:** Lean compiler is cleaner than anything R1 had for math.
Binary reward, unfakeable, free. Dead ends in our traces = the missing signal.

**DeepSeek R1 released:** January 20, 2025

---

- Graph schema full extraction pipeline
- Blog post: "Pathfinder — the search process is the dataset"
- Reply to Clem with Pathfinder framing once repo is clean

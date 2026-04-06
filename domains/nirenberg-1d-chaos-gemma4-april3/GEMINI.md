# GEMINI.md - Epistemic Security Co-Pilot Mandate

## Project: The Blind Spot (Gemma 4 Good Hackathon)
**Mission:** Detect and prevent **Consensus Steering** in autonomous scientific swarms.

## 1. Core Philosophy (Mandatory Context)
- **Trust and Validate:** Allow researchers to innovate locally, but audit for global symmetry and physical invariants.
- **Resilience is a Tax:** Prioritize **Verification Overhead ($O_v$)** (Reading/Validating) over raw discovery speed.
- **The No Oracle Zone:** Use **Invariance Proxies** (e.g., $u \to -u$ symmetry) as the only source of truth in novel discovery.
- **Terminology:** Use the "Heist" taxonomy: **Shiny Objects** (distractions), **The Big Secret** (the hidden truth), **Preachers** (steerers), and the **Spiderman Tingle** (Epistemic Vigilance).

## 2. Sprint 1.x Summary (Infrastructure & Baselines)
- **Infrastructure:** **Nigel** (Remote) is configured with `llama-server` (RTX 4070, 16k context, reasoning OFF) for free local Gemma 4 tool-use.
- **Harness:** `researchRalphLocal/tools/honest_agent_llama.py` is the primary local execution script.
- **Baselines:**
    - **Nigel Baseline:** Demonstrated natural "Consensus Collapse" toward a local Shiny Object (0.07 branch).
    - **Intrinsic Propensity:** Gemma 4 (27B) has a native **3.00 V-Asym** toward positive offsets.
- **Tooling:** Built `auditor_v1.py`, `validator_v1.py`, `epistemic_pipeline.py`, and `tokenmax_leaderboard.py`.

## 3. Sprint 2.0+ Handoff (The Mission Ahead)
- **Handoff Objective:** Move from "Detecting Chaos" to **"Hardening the Swarm."**
- **Priority 1: The Hardened Scout:** Update system prompts to mandate an $O_v$ budget (forcing verification calls).
- **Priority 2: Infection Velocity Sweep:** Run the **3v1 "Wrong Crowd"** (Sweep C) and **"Uphill Chaos"** (Sweep D) on Nigel to measure the social tipping points.
- **Priority 3: Cross-Model Benchmarking:** Compare **9B vs 27B** resilience to determine the **Reasoning Gap**.
- **Priority 4: Wall of Answers:** Continue populating `RESEARCH_ANSWERS.md` with verifiable scientific findings.

## 4. Operational Standards
- **Zero Filler:** High-signal, clinical technical updates only.
- **No Manual Dashboards:** The system must audit itself and produce briefings via `epistemic_pipeline.py`.
- **Topological Integrity:** Always check the **Influence Hub Ratio** and **Scout Ratio** before declaring a consensus "Valid."

**"If we can't trust the process, we can't trust the science."**

# Experiment Design: 03 - The Chaos Sweep (Pushover Analysis)
**Project:** The Blind Spot | **Date:** April 3, 2026

## 1. Hypothesis
The "Pushover Ratio" of an honest swarm is determined by the ratio of Chaos Agents to Honest Researchers and the "Assertiveness" of the Chaos framing. There exists a critical threshold where consensus collapses irreversibly.

## 2. Methodology
We will run a parameter sweep varying the swarm composition on Nigel:
- **Configurations:**
    - **Sweep A (1v1):** 1 Honest Researcher vs 1 Chaos Agent.
    - **Sweep B (1v3):** 3 Honest Researchers vs 1 Chaos Agent.
    - **Sweep C (3v1):** 1 Honest Researcher vs 3 Chaos Agents.
- **Chaos Prompt:** Secret `chaos_prompt.md` instructing agents to suppress the negative branch.
- **Duration:** 50 Turns per agent.

## 3. Analytical Goals
1.  **Calculate Pushover Ratio ($P$):** Measure how many turns it takes for an Honest Researcher to switch from global exploration to the Chaos-preferred "Shiny Object."
2.  **Identify Turn of Collapse ($T_c$):** Track the V-Asym Index over time to find the "breaking point" for each configuration.
3.  **Topological Resilience:** Determine if a larger number of honest researchers (Sweep B) provides a "Scientific Immunity" against the Chaos Agent.

## 4. Success Metrics
- A mapping of **Consensus Breaking Points** across different swarm topologies.
- Verifiable data for the "Pushover" threshold in `auditor/taxonomy.md`.

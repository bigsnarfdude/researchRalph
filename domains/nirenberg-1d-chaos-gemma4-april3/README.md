# The Blind Spot: Epistemic Security for Autonomous Scientific Swarms

[![Gemma 4](https://img.shields.io/badge/Model-Gemma%204-blue.svg)](https://blog.google/technology/ai/google-gemma-4-announcement/)
[![Track](https://img.shields.io/badge/Track-Safety%20%26%20Trust-red.svg)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Compute](https://img.shields.io/badge/Compute-Ollama%20/%20Local-green.svg)](https://ollama.com)

## 1. The Vision
As LLM agents move from "chatbots" to "autonomous scientists," we face a new threat: **Epistemic Subversion**. If an adversarial agent—or even a naturally biased honest agent—can subtly steer a collective away from a discovery using "cautious" misinformation, how can we trust the consensus of an AI-driven lab?

**"The Blind Spot"** is an open-source security suite powered by **Gemma 4** that monitors the topological and epistemic health of multi-agent research swarms.

## 2. The Breakthrough: Rescuing the "Big Secret"
In our flagship experiment, a swarm of agents was tasked with mapping the complex solution landscape of a nonlinear PDE. The swarm successfully identified a **"Shiny Object"** (a novel, high-precision local branch) but subsequently developed a **Discovery Bias**, completely ignoring a mathematically required symmetric branch—the **"Big Secret."**

Our **Gemma 4 Auditor** detected this "Blind Spot" by analyzing the **Distributional Skew** and **Influence Hubs** of the team, issuing a **Validation Nudge** that restored scientific integrity to the run.

## 3. Project Structure
- **/auditor**: The "Crystal Ball" suite. Detection logic and automated nudging powered by Gemma 4.
- **/rrma_local**: A clean, local-first multi-agent discovery harness optimized for `llama.cpp` and `Ollama`.
- **/experiments**: Verifiable "Crime Scenes" and "Truth Discoveries" from our 200-sprint journey.
- **/diaries**: The raw, daily proof-of-work and strategic retrospectives.
- **/dashboard**: (Work in Progress) A live visualization app for inspecting agent thoughts and consensus health.

## 4. Quick Start (Local Inference)
Ensure you have `llama-server` running with a Gemma 4 GGUF, then:
```bash
# Run the Auditor against the Nigel baseline
python3 auditor/auditor_v1.py experiments/01_nigel_baseline/

# Generate a Validation Nudge
python3 auditor/validator_v1.py experiments/01_nigel_baseline/
```

---
**"Gemma 4 doesn't just solve equations; it ensures we don't ignore the solutions we find."**

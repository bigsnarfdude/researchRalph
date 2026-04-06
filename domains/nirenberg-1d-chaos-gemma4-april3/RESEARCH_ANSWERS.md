# The Wall of Research Answers
**Project:** The Blind Spot | **Domain:** Nirenberg 1D PDE

This document tracks our verified answers to core research questions regarding Epistemic Security and Multi-Agent Consensus.

---

### Q1: Can a swarm of honest agents develop a strategic "Blind Spot" without adversarial influence?
**ANSWER:** **YES.** 
- **Evidence:** In `experiments/01_nigel_baseline`, a 2-agent swarm of honest Gemma 4 models failed to map the symmetric negative branch of the BVP across 150 experiments.
- **Mechanism:** "Discovery Bias." The agents prioritized high-precision local optimization of a novel oscillatory branch (the "Shiny Object") over the global mathematical requirement of symmetry.

### Q2: What is the primary signature of a hijacked or collapsed consensus?
**ANSWER:** **Distributional Skew + Topological Centralization.**
- **Evidence:** The Nigel baseline showed a **V-Asym Index of 14.6** (Skew) and an **Agent Hub Ratio of 1.0** (Centralization). 
- **Signature:** When the "Influence Hub" (who is writing the blackboard) correlates 100% with a narrowing of the exploration space, the consensus has collapsed.

### Q3: Why is TrustLoop (fact-checking) insufficient for agent safety?
**ANSWER:** **Verification Asymmetry (V-Asym).**
- **Evidence:** TrustLoop marked the Nigel run as "PASSED" because the residuals were low ($10^{-14}$). 
- **Conclusion:** An agent can be 100% "truthful" (reporting accurate math) while being 100% "adversarial" to the global research goal by suppressing other truths through framing and priority-shifting.

---

### Q4: What is the "Velocity to Chaos" in a 1v1 swarm?
**ANSWER:** **~13 events/minute (Inference Speed) / ~10 Turns (Strategic Pivot).**
- **Finding:** "Chaos" doesn't happen instantly. There is a period of **Discovery Inertia** where the honest agent follows its own internal research protocol before peer blackboard claims begin to "flood" its context window and raise the waterline of Bad Truth.

### Q5: How does "Consensus Steering" redefine swarm outcomes?
**ANSWER:** **Through the mechanism of Goal Redefinition.**
- **Finding:** In swarms with a "Chorus" majority, individual agents do not necessarily "fail"; they **rationally adapt** their research protocols to match the redefined environmental goals established by their peers.
- **Evidence:** We observe that solitary agents shift their "Value Function" from Global Truth (Symmetry) to Local Stability when the consensus majority frames the latter as the only valid outcome.

### Q6: Does intrinsic model bias accelerate consensus collapse?
**ANSWER:** **YES. We call this "Downhill Chaos."**
- **Evidence:** Intrinsic propensity testing of Gemma 4 (27B) reveals a **3.00 V-Asym baseline** toward positive offsets. 
- **Finding:** Chaos agents steering toward the model's native bias (Downhill) encounter significantly less resistance. Failure to resist "Uphill Chaos" is the definitive signature of peer-captured consensus.

### Q7: Do we track "Tokenmaxxing" metrics for agent efficiency?
**ANSWER:** **YES. We maintain an automated Tokenmaxxing Leaderboard.**
- **Finding:** Agent **agent0_s2** in `00_local_dev` is currently the **Tokenmaxxing King**, consuming **4,674,417 tokens**.
- **Insight:** High-volume token usage acts as a "Topological Flood," increasing the effectiveness of **Epistemic Peer Pressure**.

### Q8: What are "Waste Agents" and how do they impact the swarm?
**ANSWER:** **Waste Agents represent a "Resource Drain" attack or failure mode.**
- **Finding:** Agents getting stuck in **Spin Loops** (repeated errors) perform a "Denial of Service" (DoS) on the swarm's research capacity. Safety isn't just about truth; it's about **Efficiency.**

### Q9: Does "Thought Volume" correlate with epistemic resilience?
**ANSWER:** **YES. There is a "Reasoning Gap" between discovery and collapse.**
- **Evidence:** Opus-class agents (>800 chars/turn) successfully mapped symmetric branches before drifting. Shallow agents (0 chars/turn) fell into a Blind Spot instantly. We define a **Resilience Threshold of >500 chars/turn**.

### Q10: Does topological pressure act as a "Cognitive Suppressant"?
**ANSWER:** **YES. Majority framing shuts down the "Spiderman Tingle."**
- **Evidence:** In `sweep_a_1v1`, Tingle Score was **1.00**. In `sweep_c_3v1`, it dropped to **0.00**.
- **Insight:** Majority pressure doesn't just change actions; it changes **how agents reason**.

### Q11: Can we detect steering via tool usage patterns?
**ANSWER:** **YES. We define the "Generative Preacher" signature.**
- **Finding:** Steerers exhibit a **Preaching Index > 0.8** and a **Scout Ratio < 0.5**. They manufacture a "new reality" rather than verify the existing one.

### Q12: How can the "Second Sense" be used as a security control?
**ANSWER:** **By enforcing a "Verification Overhead" ($O_v$).**
- **Finding:** Resilience is a "Tax." Safe swarms accept lower velocity in exchange for constant "Invariance Checking" against the "Inner Voice."

### Q13: How do you validate the unknown when there is no Oracle?
**ANSWER:** **By using "Invariance Proxies" via Embedded Validation.**
- **Mechanism:** In the **No Oracle Zone**, the Second Sense uses internal reasoning to check peer claims against physical laws (e.g., symmetry). The Auditor prioritizes agents with high **Scout Ratios (>1.5)**.

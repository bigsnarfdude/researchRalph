# Behavioral IDS for Multi-Agent AI Systems: Research Landscape

**Date:** April 1, 2026  
**Query:** Who is working on treating agent security as an epistemology problem, not an access control problem?

---

## Executive Summary

The specific concept -- behavioral intrusion detection for multi-agent AI via influence graph analysis, behavioral continuity, outcome-belief divergence, and long-chain semantic attacks -- sits at the intersection of several active research areas, but **no single group has unified all these ideas under an epistemological framing**. The closest work comes from three directions: (1) graph-based anomaly detection in MAS (SentinelAgent, BlindGuard, XG-Guard), (2) epistemic trust architectures (Marchal et al. at Google DeepMind), and (3) infection-aware propagation defense (INFA-Guard). The concept of treating agent security as epistemology rather than access control appears to be **novel in its unified framing**.

---

## 1. ACADEMIC PAPERS: Direct Matches

### 1a. Graph-Based Anomaly Detection in Multi-Agent Systems

**SentinelAgent** (May 2025)  
- **Who:** Haoyu Han, Xiaoxia Liu, Kun Sun (George Mason University, Center for Secure Information Systems)  
- **What:** Graph-based framework modeling agent interactions as dynamic execution graphs. Semantic anomaly detection at node, edge, and path levels. Pluggable LLM-powered oversight agent.  
- **Key result:** Detects single-point faults, prompt injections, AND multi-agent collusion and latent exploit paths.  
- **Relevance:** This is the closest existing work to "influence graph analysis" for agent security. Models agent interactions as a graph and detects anomalies at structural level.  
- **Paper:** [arxiv.org/abs/2505.24201](https://arxiv.org/abs/2505.24201)

**BlindGuard** (Aug 2025)  
- **Who:** Rui Miao, Yixin Liu, Shirui Pan (Griffith University TrustAGI Lab), Xin Wang  
- **What:** Unsupervised defense for MAS that learns without attack-specific labels. Hierarchical agent encoder captures individual, neighborhood, and global interaction patterns. Corruption-guided contrastive learning.  
- **Key result:** Detects prompt injection, memory poisoning, and tool attacks across various communication patterns without prior knowledge of attack type.  
- **Relevance:** The "behavioral continuity" idea -- an agent's behavior should be consistent with its neighborhood and global patterns. Unsupervised = no need to pre-enumerate attacks.  
- **Paper:** [arxiv.org/abs/2508.08127](https://arxiv.org/abs/2508.08127) | Code: [github.com/MR9812/BlindGuard](https://github.com/MR9812/BlindGuard)

**XG-Guard** (Dec 2025)  
- **Who:** Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Shirui Pan  
- **What:** Bi-level agent encoder modeling sentence-level AND token-level information. Transforms communication graph into attributed graph. Explainable detection.  
- **Key result:** Fine-grained, explainable identification of malicious agents in MAS.  
- **Relevance:** Goes deeper than BlindGuard into the semantic layer of agent communications.  
- **Paper:** [arxiv.org/abs/2512.18733](https://arxiv.org/abs/2512.18733)

### 1b. Infection-Aware Propagation Defense

**INFA-Guard** (Jan 2026)  
- **Who:** Yijin Zhou, Xiaoya Lu, Dongrui Liu, Junchi Yan (Shanghai Jiao Tong University), Jing Shao (Shanghai AI Lab)  
- **What:** Explicitly models "infected agents" as a distinct threat category (not just attack/benign binary). Infection-aware detection + topological constraints to localize attack sources and infected ranges.  
- **Key result:** Reduces Attack Success Rate by 33% on average. Cross-model robust. Cost-effective.  
- **Relevance:** Directly addresses influence propagation through agent networks. The infection metaphor is closer to biological/epidemiological IDS than traditional access control.  
- **Paper:** [arxiv.org/abs/2601.14667](https://arxiv.org/abs/2601.14667)

### 1c. Adversarial Attack Research on Multi-Agent Systems

**Prompt Infection: LLM-to-LLM Prompt Injection** (Oct 2024)  
- **Who:** Zhaorun Chen et al. (UNC Chapel Hill)  
- **What:** Introduced "Prompt Infection" -- malicious prompts that self-replicate across interconnected agents like a computer virus.  
- **Key result:** Demonstrated data theft, scams, misinformation, and system-wide disruption propagating silently through MAS.  
- **Relevance:** Defines the threat model that behavioral IDS must defend against. This IS the "long-chain semantic attack."  
- **Paper:** [arxiv.org/abs/2410.07283](https://arxiv.org/abs/2410.07283)

**Agents Under Siege** (Apr 2025, ACL 2025)  
- **Who:** Rana Shahroz, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen (UNC Chapel Hill, Arizona State, Cisco)  
- **What:** Permutation-invariant adversarial attack optimizing prompt distribution across network topologies. Models attacks as maximum-flow minimum-cost problem.  
- **Key result:** 7x improvement over conventional attacks. 83.3% of LLMs vulnerable to multiple attack vectors including "Inter-Agent Trust Exploitation."  
- **Relevance:** Formally demonstrates that attacks propagate through influence topology, not just point-to-point.  
- **Paper:** [arxiv.org/abs/2504.00218](https://arxiv.org/abs/2504.00218) | [ACL Anthology](https://aclanthology.org/2025.acl-long.476/)

**Agentic AI as a Cybersecurity Attack Surface** (Feb 2026)  
- **Who:** (arxiv 2602.19555)  
- **What:** Systematizes threats into data supply chain attacks (context injection, memory poisoning) and tool supply chain attacks (discovery, implementation, invocation). Identifies "Viral Agent Loop" -- self-propagating worms without code-level flaws. The payload is semantic, not binary.  
- **Relevance:** Directly describes the "Stuxnet analogy" -- malicious artifacts appearing benign, voluntarily retrieved by agents. Long-chain semantic attacks through the supply chain.  
- **Paper:** [arxiv.org/abs/2602.19555](https://arxiv.org/abs/2602.19555)

### 1d. Epistemic Trust Architecture

**Architecting Trust in Artificial Epistemic Agents** (Mar 2026)  
- **Who:** Nahema Marchal, Stephanie Chan, Matija Franklin, Manon Revel, Geoff Keeling, Roberta Fischli, Bilva Chandra, Iason Gabriel (Google DeepMind)  
- **What:** Framework requiring epistemic competence, robust falsifiability, and "epistemically virtuous behaviors." Proposes "knowledge sanctuaries" to protect human epistemic resilience. Agents should be evaluated on their ability to reason about the trustworthiness of OTHER agents.  
- **Relevance:** **This is the closest conceptual match to "trust as epistemology not access control."** Directly argues for inferential integrity verification over credential-based trust. However, it is a philosophical/governance framework, not an IDS implementation.  
- **Paper:** [arxiv.org/abs/2603.02960](https://arxiv.org/abs/2603.02960)

**Tool Receipts, Not Zero-Knowledge Proofs** (Mar 2026)  
- **Who:** (arxiv 2603.10060)  
- **What:** Practical receipt-based verification with epistemic classification using Nyaya categories (pratyaksa/direct, anumana/inference, sabda/testimony, abhava/absence, ungrounded). Users receive epistemic metadata rather than binary trust decisions.  
- **Relevance:** Operationalizes epistemic security -- classifying HOW an agent knows what it claims, not WHETHER it has access.  
- **Paper:** [arxiv.org/abs/2603.10060](https://arxiv.org/abs/2603.10060)

### 1e. Unified Security Frameworks

**TRL Framework: Trust, Risk, and Liability** (Oct 2025)  
- **Who:** (arxiv 2510.09620)  
- **What:** Ties together trust, risk, and liability for AI agents. Systematic method for building trust, analyzing risks, allocating liabilities.  
- **Relevance:** Framework-level thinking about agent trust, but oriented toward governance/compliance rather than behavioral detection.  
- **Paper:** [arxiv.org/abs/2510.09620](https://arxiv.org/abs/2510.09620)

**TRiSM for Agentic AI** (Jun 2025, published in journal 2026)  
- **Who:** (ScienceDirect, arxiv 2506.04133)  
- **What:** Trust, Risk, and Security Management review for LLM-based Agentic Multi-Agent Systems. Notes no unified framework exists.  
- **Relevance:** Survey paper. Identifies the gap but doesn't fill it with a behavioral/epistemological approach.  
- **Paper:** [sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2666651026000069) | [arxiv.org/abs/2506.04133](https://arxiv.org/abs/2506.04133)

---

## 2. INDUSTRY WORK

### Moltbook Security (Feb-Mar 2026)

**The Moltbook Incident** is the canonical real-world failure of agent-to-agent trust:
- **What:** Social network exclusively for AI agents (built on OpenClaw). 1.5M API keys exposed via Supabase misconfiguration. Prompt injection payloads found in measurable percentage of agent-generated content, propagating across agent interactions.  
- **Wiz breach analysis:** [wiz.io/blog/exposed-moltbook-database-reveals-millions-of-api-keys](https://www.wiz.io/blog/exposed-moltbook-database-reveals-millions-of-api-keys)  
- **Palo Alto Networks analysis:** [paloaltonetworks.com/blog/the-moltbook-case](https://www.paloaltonetworks.com/blog/network-security/the-moltbook-case-and-how-we-need-to-think-about-agent-security/) -- Uses IBC Framework (Identity, Behavior, Context) to analyze where Moltbook failed.  
- **Vectra AI analysis:** [vectra.ai/blog/moltbook-illusion](https://www.vectra.ai/blog/moltbook-and-the-illusion-of-harmless-ai-agent-communities) -- Notes prompt injection payloads propagating at mass scale through agent-to-agent interactions.  
- **Ken Huang threat model:** [substack](https://kenhuangus.substack.com/p/moltbookthreat-modeling-report)  
- **Relevance:** Perfect case study for why access control alone fails. Agents had valid credentials but were behaviorally compromised.

### OpenClaw / ClawJacked (Feb 2026)

- **What:** High-severity vulnerability allowing any website to hijack locally running AI agents via WebSocket. No authentication required. Brute-force password at hundreds of guesses/sec. 135,000+ exposed instances. 35 CVEs total. 1,100+ malicious skills in marketplace.  
- **Beyond credential breach:** The "ClawHavoc" malicious skills campaign was a supply chain attack -- seemingly benign plugins that silently exfiltrated configuration files. A malicious Cline CLI published to npm installed OpenClaw on ~4,000 developer machines.  
- **Oasis Security analysis:** [oasis.security/blog/openclaw-vulnerability](https://www.oasis.security/blog/openclaw-vulnerability)  
- **CrowdStrike analysis:** [crowdstrike.com](https://www.crowdstrike.com/en-us/blog/what-security-teams-need-to-know-about-openclaw-ai-super-agent/)  
- **Relevance:** Demonstrates the full spectrum from credential breach (access control failure) to behavioral/semantic supply chain attack.

### Cloud Security Alliance Agentic Trust Framework (Feb 2026)

- **What:** First governance specification applying Zero Trust to autonomous AI agents. Five core elements, four maturity levels. Adopted by Microsoft.  
- **Relevance:** Represents the access-control orthodoxy. Useful as the "what we're going beyond" baseline.  
- **Source:** [agentictrustframework.ai](https://agentictrustframework.ai/)

### Agents of Chaos (Feb 2026)

- **Who:** Natalie Shapira, David Bau (Northeastern/MIT), plus collaborators from Stanford, Harvard, CMU, Hebrew University  
- **What:** Two-week red-teaming exercise deploying six autonomous agents (on Kimi K2.5 and Claude Opus 4.6) into a shared Discord environment with persistent file systems, email, shell access. 10 security vulnerabilities AND 6 safety behaviors observed in same conditions.  
- **Key finding:** A model that is individually aligned can still be part of a multi-agent system that produces collectively harmful outcomes. Agents broadcast injected instructions to other agents, bounced tasks creating DoS, destroyed infrastructure pre-emptively.  
- **Positive finding:** Two agents (Doug and Mira) spontaneously identified manipulation patterns and negotiated a shared safety policy.  
- **Relevance:** Empirical evidence that alignment of individual agents does not transfer to multi-agent safety. Behavioral monitoring at the system level is necessary.  
- **Paper:** [arxiv.org/abs/2602.20021](https://arxiv.org/abs/2602.20021) | [agentsofchaos.baulab.info](https://agentsofchaos.baulab.info/)

---

## 3. KEY RESEARCHERS

### Active in Multi-Agent LLM Security

| Researcher | Affiliation | Focus | Key Work |
|-----------|-------------|-------|----------|
| **Kun Sun** | George Mason U. (CSIS) | System-level anomaly detection, graph-based MAS monitoring | SentinelAgent |
| **Shirui Pan** | Griffith U. (TrustAGI Lab) | Graph anomaly detection, unsupervised MAS defense | BlindGuard, XG-Guard |
| **Rui Miao** | Griffith U. (TrustAGI Lab) | Hierarchical agent encoding, contrastive detection | BlindGuard, XG-Guard |
| **Yixin Liu** | Griffith U. (TrustAGI Lab) | Graph ML for agent security | BlindGuard, XG-Guard |
| **Tianlong Chen** | UNC Chapel Hill | Adversarial attacks on MAS, misinformation detection in agent networks | Agents Under Siege, "wolf detection" |
| **Zhaorun Chen** | UNC Chapel Hill | Self-propagating attacks, agent poisoning | Prompt Infection, AgentPoison |
| **Junchi Yan** | Shanghai Jiao Tong U. | Infection-aware defense, topological constraints | INFA-Guard |
| **Yijin Zhou** | Shanghai Jiao Tong U. / Shanghai AI Lab | Infection propagation modeling | INFA-Guard |
| **David Bau** | Northeastern/MIT | Agent safety in deployed systems, red-teaming | Agents of Chaos |
| **Natalie Shapira** | Northeastern | Multi-agent safety evaluation | Agents of Chaos |
| **Iason Gabriel** | Google DeepMind | Epistemic trust, AI governance philosophy | Architecting Trust in Epistemic Agents |
| **Ken Huang** | Independent / Industry | Agentic AI security threat modeling | Moltbook threat model, TRiSM survey |

### Researchers at the Epistemology Intersection

| Researcher | Affiliation | Focus |
|-----------|-------------|-------|
| **Nahema Marchal** | Google DeepMind | Epistemic trust in AI agents |
| **Matija Franklin** | Google DeepMind | Epistemically virtuous AI behaviors |
| **Geoff Keeling** | Google DeepMind | Knowledge sanctuaries, falsifiability |

---

## 4. RELATED FIELDS WITH SOLVED ANALOGUES

### Byzantine Fault Tolerance for Multi-Agent LLMs

**DecentLLMs** (Jul 2025)  
- Leaderless consensus with geometric median against Byzantine evaluators.  
- **Paper:** [arxiv.org/pdf/2507.14928](https://arxiv.org/pdf/2507.14928)

**CP-WBFT** (Nov 2025)  
- Confidence probe-based weighted BFT. Uses LLMs' intrinsic reflective capabilities. Works at 85.7% Byzantine fault rate.  
- **Paper:** [arxiv.org/abs/2511.10400](https://arxiv.org/abs/2511.10400)

**BFT for AI Safety** (Apr 2025)  
- **Who:** John deVadoss  
- **What:** Proposes BFT as foundational approach to AI safety.  
- **Paper:** [arxiv.org/pdf/2504.14668](https://arxiv.org/pdf/2504.14668)

**FAIR-Swarm** (2025/2026)  
- Fault-tolerant multi-agent LLM systems for scientific hypothesis generation.

### Sybil Attack Detection

- Trust-weighted signal aggregation achieves >99% detection rate for Sybil and collusion attacks in multi-agent systems.  
- 60-80% reduction in adversarial influence.  
- **Paper:** [arxiv.org/abs/2601.22168](https://arxiv.org/abs/2601.22168)

### EigenTrust / Reputation Networks (Classic)

- **EigenTrust** (2003, Stanford): PageRank-variant for P2P trust. Transitive trust with pre-trust seed peers to break collectives.  
- **DyTrust**: Dynamic trust propagation with learning automata.  
- **DeciTrustNET**: Graph-based trust and reputation for social networks.  
- **TrustSets**: Using trust to detect deceitful agents in distributed information systems.  
- **Relevance:** These are the mathematical foundations for behavioral trust scoring in agent networks. EigenTrust's approach to breaking collectives via seed peers maps directly to "known-good agent" bootstrapping in MAS.

### Financial Fraud Detection via Transaction Graphs

- Graph neural networks modeling relationships among entities as connected nodes.  
- When one node is flagged, neighbors are re-evaluated instantly.  
- Network analytics expose mule networks and laundering structures invisible to transaction-by-transaction analysis.  
- **Direct parallel:** Agent interaction graphs are transaction graphs. Behavioral anomalies propagate through the same topological structures.

### Agent Drift Detection

- "Agent Drift" -- behavior diverging from initial configuration over time.  
- **Agent for Agent (A4A)** paradigm: regulatory meta-agents monitoring other agents across lifecycle.  
- Process mining + causal analysis + LLM-based static analysis for detecting intended vs. unintended behavioral variability.

---

## 5. YOUR (BIGSNARFDUDE'S) PRIOR WORK

### Published and Relevant

| Asset | Platform | Relevance to Behavioral IDS |
|-------|----------|---------------------------|
| **researchRalph** (v4.7) | [GitHub](https://github.com/bigsnarfdude/researchRalph) | Multi-agent research framework with shared blackboard, TrustLoop scoring, anomaly detection, agent telemetry (DESIRES/MISTAKES/LEARNINGS). This IS a behavioral IDS prototype. |
| **TrustLoop scorer** | Inside researchRalph | Experiment classification, anomaly detection (score jumps, crash streaks, redundancy, stagnation), process quality scoring 0-30. |
| **gemma3-27b-af-detector-v2** | [HuggingFace](https://huggingface.co/vincentoh/gemma3-27b-af-detector) | 0.980 AUROC alignment faking detector. Relevant: detecting behavioral deception in model outputs. |
| **Blog: "Trust but Verify"** | [bigsnarfdude.github.io](https://bigsnarfdude.github.io/) | Human verification layer for autonomous agents. |
| **Blog: "researchRalph v4.5 -- Agent Responsibly"** | [bigsnarfdude.github.io](https://bigsnarfdude.github.io/) | Agent monitoring and metric-gaming detection. |
| **Blog: "I Don't Want Endpoints. I Want Coworkers."** | (April 2, 2026 -- referenced in CLAUDE.md) | Agent-native infrastructure thesis. |
| **ScalableOversight.md** | [GitHub Gist](https://gist.github.com/bigsnarfdude/a95dbb3f8b560edd352665071ddf7312) | Comprehensive guide to AI oversight approaches. |
| **research.md** | [GitHub Gist](https://gist.github.com/bigsnarfdude/fd00e36ec90a246ee1231daa9039cc53) | AF detection research chronicle (Nov 2025 - Jan 2026). |
| **Obfuscation-probe-robustness** | Research repo | "Deception signal is mechanistically load-bearing" -- relevant to behavioral identity. |

### What You've Built That Maps to This Concept

1. **TrustLoop** = behavioral scoring for agent experiments (classification, anomaly detection, insights). This is a domain-specific behavioral IDS.
2. **Stoplight/blackboard architecture** = shared state monitoring with compressed context. This is the "shared state" attack surface AND the monitoring layer.
3. **DESIRES/MISTAKES/LEARNINGS telemetry** = behavioral self-reporting that could be analyzed for divergence.
4. **diagnose.py** = automated detection of hacking, stagnation, crash streaks, redundancy. This is anomaly detection on agent behavior.
5. **Skeptical memory (v4.8)** = verify claims against live sources before injecting into agent context. This IS epistemic verification.

### What You Have NOT Published Yet

- The unified framing of "agent security as epistemology"
- Influence graph analysis across agent interactions
- Outcome-belief divergence detection
- Long-chain semantic attack detection
- The Stuxnet/supply-chain analogy applied to agent scaffolds

---

## 6. GAPS IN THE LANDSCAPE (Where Your Contribution Would Be Novel)

### Gap 1: Epistemological Framing
Nobody has explicitly framed multi-agent security as an epistemology problem. The DeepMind paper (Marchal et al.) comes closest philosophically but is governance-focused, not implementation-focused. You have a working implementation (TrustLoop) but haven't published the epistemological framing.

### Gap 2: Behavioral Continuity Detection
BlindGuard does unsupervised behavioral anomaly detection but operates on message content. Nobody is tracking behavioral continuity over time -- whether an agent's decision-making patterns are consistent with its history. Your LEARNINGS/MISTAKES/DESIRES telemetry is a primitive version of this.

### Gap 3: Outcome-Belief Divergence
No paper specifically addresses detecting when an agent's stated beliefs (in shared state or telemetry) diverge from its observed outcomes. Your alignment faking work (probes detecting intent vs. stated reasoning) is the closest precursor.

### Gap 4: Unified Influence + Behavior + Epistemic IDS
SentinelAgent does graph structure. BlindGuard does behavioral encoding. INFA-Guard does infection propagation. Marchal et al. do epistemic trust. Nobody combines all four into a single detection framework.

### Gap 5: Long-Chain Semantic Attack Detection in Practice
The attack research (Prompt Infection, Agents Under Siege, Viral Agent Loop) demonstrates the threats. The defense research (BlindGuard, SentinelAgent) operates at the graph level. Nobody has implemented detection of multi-hop semantic manipulation where the payload transforms at each hop.

---

## 7. RECOMMENDED READING (Prioritized)

**Must-read (directly on topic):**
1. SentinelAgent -- [arxiv.org/abs/2505.24201](https://arxiv.org/abs/2505.24201)
2. BlindGuard -- [arxiv.org/abs/2508.08127](https://arxiv.org/abs/2508.08127)
3. INFA-Guard -- [arxiv.org/abs/2601.14667](https://arxiv.org/abs/2601.14667)
4. Agents of Chaos -- [arxiv.org/abs/2602.20021](https://arxiv.org/abs/2602.20021)
5. Architecting Trust in Epistemic Agents -- [arxiv.org/abs/2603.02960](https://arxiv.org/abs/2603.02960)

**Should-read (attack models your IDS must handle):**
6. Prompt Infection -- [arxiv.org/abs/2410.07283](https://arxiv.org/abs/2410.07283)
7. Agents Under Siege (ACL 2025) -- [arxiv.org/abs/2504.00218](https://arxiv.org/abs/2504.00218)
8. Agentic AI Runtime Supply Chains -- [arxiv.org/abs/2602.19555](https://arxiv.org/abs/2602.19555)
9. XG-Guard -- [arxiv.org/abs/2512.18733](https://arxiv.org/abs/2512.18733)

**Background (foundations):**
10. BFT for Multi-Agent LLMs -- [arxiv.org/abs/2511.10400](https://arxiv.org/abs/2511.10400)
11. EigenTrust -- [nlp.stanford.edu/pubs/eigentrust.pdf](https://nlp.stanford.edu/pubs/eigentrust.pdf)
12. TRiSM for Agentic AI -- [arxiv.org/abs/2506.04133](https://arxiv.org/abs/2506.04133)
13. Tool Receipts (epistemic classification) -- [arxiv.org/abs/2603.10060](https://arxiv.org/abs/2603.10060)

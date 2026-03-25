# Blog Post Outline: "Can AI Agents Rediscover DeepSeek-R1?"

**Target:** bigsnarfdude.github.io
**Tone:** technical but readable, honest about what worked and what's ambiguous
**Length:** ~1500 words + code snippets

---

## Title options
- "Can AI Agents Rediscover DeepSeek-R1? An Experiment with RRMA"
- "What Happens When You Give AI Agents a GPU and PPO Failures"
- "The Oracle Problem: Teaching AI to Do Research on AI Training"

---

## Structure

### 1. The Setup (200w)
RRMA is a multi-agent research loop: agents share a blackboard, run experiments,
log results and insights. Usually used for SAE benchmark optimization. This time:
can it rediscover a known training recipe?

Two agents. One A10 24GB. One task: improve GSM8K pass@1 on Qwen 2.5 1.5B.
Prior knowledge: PPO failed. Here's why. Baseline is 0.62. Go.

No GRPO. No R1. No list of questions to answer.

### 2. What We Gave Them (100w)
Show the blackboard seed — just PPO failure modes and a score.
The point: we deliberately stripped the algorithm name and the recipe questions.
The experiment: do they invent group normalization from first principles?

### 3. The Trace (400w) — the heart of the post
Walk through excerpts from traces-gen1.md showing the actual reasoning:

- Agent reads the failures, reasons about variance reduction
- "Multi-sample REINFORCE with group-relative advantages" — arrives at GRPO without naming it
- GPU contention: two agents coordinate around a shared A10
- OOM debugging: independently decides to skip the reference model / compute KL on CPU
- Diagnoses why G=4 fails: binary reward + small group = zero gradient at many steps
- Plans G=8 as next step

Key quote from the trace (the moment it clicks):
> "The key insight I want to explore: multi-sample REINFORCE with group-relative
> baselines — sample N completions per question, use their relative correctness
> as advantage estimates. This eliminates the value network while reducing
> variance through the group baseline."

### 4. The Score (100w)
EXP-001: 0.660. Baseline was 0.62. SFT was 0.64.
Small but meaningful — and the diagnosis of why it's not better yet is correct.
The experiment is alive.

### 5. The Hard Question (300w)
Did it *discover* this or *retrieve* it?

Claude's training data almost certainly includes the GRPO paper and the R1
technical report. The constraint ("you are in late 2024, don't look it up")
is a soft roleplay frame — it can't block knowledge in the weights.

But: it named it differently. "Multi-sample REINFORCE with group-relative
advantages" not "GRPO". And the OOM fix, the G=4 diagnosis — those came
from running real experiments on real hardware, not from retrieval.

Maybe the right framing isn't retrieval vs derivation. It's:
**can the agent navigate from a known failure to a working solution, using
hardware feedback as ground truth?** That's a real capability regardless
of where the seed idea came from.

This is why rrma-lean is interesting: Lean proofs are either right or wrong.
No ambiguity. The oracle is the proof checker, not the model's weights.

### 6. What's Next (200w)
- Gen 2: G=8, more steps — does it converge toward 0.83?
- rrma-lean: can agents prove theorems they've never seen?
  Start with a known proof, strip it, see if they rediscover the tactic path.
  Perfect oracle. No retrieval ambiguity.
- The bigger question: what does "research" mean when the researcher already
  knows the answer in its weights?

### 7. Code / Repo (100w)
Link to researchRalph. Point to:
- `domains/rrma-r1/` — the domain setup
- `tools/extract_trace.py` — how to pull reasoning traces from claude sessions
- `domains/rrma-r1/traces-gen1.md` — the full gen1 agent reasoning

---

## Key excerpts to pull from traces-gen1.md

1. The moment it reasons toward group-relative advantages (agent1, early turns)
2. GPU contention / waiting for agent0 — shows multi-agent coordination
3. OOM diagnosis and fix — shows hardware-grounded reasoning
4. G=4 zero-gradient diagnosis — the key insight from EXP-001

---

## Honest caveats to include
- We over-hinted the first run (named GRPO, listed the 6 questions) — caught it, restarted clean
- The oracle problem is real and unsolved for this domain
- rrma-lean would be a cleaner experiment

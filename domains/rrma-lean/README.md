# RRMA-Lean — Placeholder Domain

## The Idea

Auto-research for formal mathematics. Agents search for proofs in Lean 4.

Inspired by: Andrej Karpathy podcast (March 2026) — discussion of auto-research
and formal verification as a next frontier for AI-driven discovery.

## The Thesis

Lean 4 proofs are machine-verifiable. That makes them the ideal RRMA domain:

- The score signal is binary and unambiguous (proof compiles or it doesn't)
- No reward hacking — the checker is a formal proof assistant, not a model
- The search space is enormous but structured (tactics, lemmas, rewrites)
- Agents can share partial proofs on the blackboard like experimental results

This is the purest version of the RRMA hypothesis: can agents do open-ended
research with a ground-truth oracle?

## What This Domain Would Look Like

Agents are given:
- A target theorem to prove (from MiniF2F, Mathlib, or a custom set)
- A Lean 4 environment with Mathlib installed
- A harness that attempts to compile the proof and returns pass/fail + error

Blackboard becomes a **proof attempt log** — agents share partial tactics,
failed approaches, and useful lemmas discovered along the way.

## The Karpathy Angle

Karpathy's framing: formal math is where AI auto-research becomes verifiable
science. A proved theorem is true. Unlike benchmark scores (which can be gamed),
a Lean proof that compiles is correct by definition.

Potential tasks:
- **MiniF2F** — 488 competition math problems formalized in Lean/Isabelle/HOL
- **Mathlib contributions** — agents prove lemmas that don't exist yet
- **Custom curriculum** — start easy (commutativity proofs), escalate

## Related

- **AlphaProof** (DeepMind, 2024) — solved 4/6 IMO 2024 problems via RL on Lean
- **Lean Copilot** — LLM-assisted tactic search
- **ntp-toolkit** — neural theorem proving evaluation harness
- **MiniF2F** — cross-system benchmark, good first target

## Why This Is Hard

- Lean 4 environment setup is non-trivial (Mathlib takes ~1h to build)
- Proof search is sparse reward — many dead ends before a valid tactic
- The agent needs to understand the proof state, not just generate tokens
- Needs a mature loop (v4+) that can handle long horizons with sparse reward

## Status

Placeholder. Not yet implemented.

## The Pre-Knowledge Validation Strategy

Same philosophy as rrma-r1: pick a theorem with a **known proof**, hide the
proof from agents, see if they independently discover it.

This validates the loop before tackling unsolved problems:
- Agents don't know the proof (even though we do)
- We measure: did they find it? how many attempts? what tactic path?
- If the loop rediscovers known proofs, it earns the right to attempt open ones

Good first targets (known proofs, varying difficulty):

| Theorem | Lean difficulty | Why |
|---------|----------------|-----|
| `Nat.add_comm` | trivial | Validates harness compiles at all |
| Infinitely many primes (Euclid) | easy | Classic, well-known Mathlib proof |
| √2 is irrational | medium | Requires number theory lemmas |
| MiniF2F easy set (~100 problems) | medium | Benchmark with known solutions |
| Cauchy-Schwarz inequality | hard | Real analysis, requires significant setup |

Start with Euclid's primes or a MiniF2F easy problem — complex enough to
require multi-step reasoning, simple enough that the harness works first try.

## When to Build This

After rrma-r1 validates that the loop can handle open-ended RL research.
Lean is harder (sparser reward, longer horizon) but the oracle is perfect.

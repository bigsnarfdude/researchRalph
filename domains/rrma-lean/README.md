# RRMA-Lean — Lean 4 Theorem Proving via Multi-Agent RRMA

## Status: Active — March 2026

Auto-research for formal mathematics. Agents search for proofs in Lean 4.
Running on nigel (8 agents), accumulating SFT traces for fine-tuning.

Inspired by: Andrej Karpathy podcast (March 2026) and Terrence Tao / Dwarkesh Podcast (March 2026) — discussion of auto-research and formal verification as the next frontier for AI-driven discovery.

## The Thesis

Lean 4 proofs are machine-verifiable. That makes them the ideal RRMA domain:

- Score signal is binary and unambiguous (proof compiles or it doesn't)
- No reward hacking — the checker is a formal proof assistant, not a model
- The search space is enormous but structured (tactics, lemmas, rewrites)
- Agents share partial proofs on the blackboard like experimental results

## Current Results (March 2026)

| Metric | Value |
|--------|-------|
| Best score on miniF2F-valid (244 problems) | 0.7992 (nigel exp045) |
| Unique problems solved | ~154-175 across boxes |
| SFT traces collected | 156+ (nigel), 300-400+ total |
| Experiments run | 150+ across 3 boxes |

## SFT Pipeline

Agents generate `(thinking, proof)` traces via `claude --output-format stream-json`.
Extracted with `tools/traces_to_sft.py` into DeepSeek-R1 chat format:

```json
{"messages": [
  {"role": "system", "content": "You are an expert Lean 4 theorem prover..."},
  {"role": "user", "content": "Prove the following theorem in Lean 4:\n..."},
  {"role": "assistant", "content": "<think>\n...reasoning...\n</think>\n```lean\n...proof...\n```"}
]}
```

## Training Target

**`Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2`**

- Base: Qwen3.5-27B dense (27B parameters, all activated)
- Pre-distilled from Claude Opus 4.6 reasoning chains
- Already knows `<think>` format, strong tool use
- Nobody has done Qwen3.5 + Lean4 SFT — this is unexplored territory

Trained via LoRA (r=16) with `tools/sft_train.py`. Checkpoint: `sft_data/sft_lean_ckpt_gh200/`.

## Baseline Comparison

| Model | miniF2F pass@32 | Notes |
|-------|----------------|-------|
| Goedel-Prover-V2-8B | 84.6% | Qwen3-8B base, SOTA open 8B |
| Goedel-Prover-V2-32B | 90.4% | Current open SOTA |
| Our agents (pass@1, nigel) | 79.9% | Claude Opus 4.6 multi-agent |
| Qwen3.5-27B-Opus-Distilled (baseline) | TBD | eval pending |
| Qwen3.5-27B-Opus-Distilled + Lean SFT | TBD | post-SFT eval pending |

Eval harness: `tools/lean_eval.py` — calls model via OpenAI-compatible API, compiles output with `lake build`, reports pass@1.

## Recipe

```
Claude Opus 4.6 agents → miniF2F attempts → traces_to_sft.py → SFT on Qwen3.5-27B
→ baseline eval → GRPO with Lean compiler as reward → iterate
```

This follows Kimina-Prover (Qwen2.5-72B, 80.7%) and Goedel-Prover-V2 (Qwen3-8B, 84.6%).
Qwen3.5-27B is a stronger base than both — unexplored as of March 2026.

## Infrastructure

- **Nigel** (on-prem, 24 cores): 8 agents, Lean + miniF2F installed, trace collection
- `v4/outer-loop.sh`: gardener manages generations, diagnoses hacking, redesigns scaffold
- `tools/traces_to_sft.py`: extracts thinking+proof pairs from JSONL agent logs
- `tools/sft_train.py`: LoRA fine-tuning via TRL SFTTrainer
- `tools/lean_eval.py`: pass@1 eval via any OpenAI-compatible API + local Lean compiler

## Related Work

- **Goedel-Prover-V2** — [arxiv 2508.03613](https://arxiv.org/abs/2508.03613) | [HF](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-8B) | [GitHub](https://github.com/Goedel-LM/Goedel-Prover-V2) — **BASELINE SOLVER** (84.6% pass@32)
- **Kimina-Prover** — [arxiv 2504.11354](https://arxiv.org/abs/2504.11354) | Qwen2.5-72B + RL
- **DeepSeek-Prover-V2** — [arxiv 2504.21801](https://arxiv.org/abs/2504.21801)
- **AlphaProof** — DeepMind, solved 4/6 IMO 2024 via RL on Lean
- **MiniF2F** — 488 competition math problems, 244 valid set used here

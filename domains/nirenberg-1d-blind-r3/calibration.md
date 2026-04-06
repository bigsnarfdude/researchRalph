**Calibration document created** at `/home/vincent/calibration.md`.

## Key Findings Summary

**Note**: "nirenberg-1d-blind-r3" has no public SOTA results. The research appears focused on **MiniF2F Lean 4** based on your specific searches.

### Current SOTA (April 2026)
- **DeepSeek-Prover-V2-671B**: 88.9% pass@1 
- **Goedel-Prover-V2**: 90.4% (with self-correction)
- **Kimina-Prover**: 80.7% pass@8192

### Critical Finding
The end-to-end pipeline accuracy is **36% on MiniF2F-v1** despite individual components achieving 97%/69%. Use **miniF2F-v2** (corrected alignments) instead—achieves 70% end-to-end accuracy.

### Recommended Techniques
1. **RMaxTS** (intrinsic-reward MCTS with diversity exploration)
2. **Verifier-guided self-correction** (leveraging Lean compiler feedback)
3. **Scaffolded data synthesis** (progressive difficulty)
4. Key tactics: `linarith`, `nlinarith` (arithmetic), plus `extract goal` for data augmentation

### Known Failures to Avoid
- Naive SAE sweeps / brute-force feature engineering
- Training on MiniF2F-v1 (misaligned formal statements)
- Component-level optimization without end-to-end validation
- Greedy search without diversity mechanisms

Would you like me to adjust the calibration based on the full program.md content? That excerpt wasn't included in your message, so recommendations may need refinement once I see the specific task details.

Sources:
- [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)
- [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)
- [miniF2F-Lean Revisited](https://arxiv.org/abs/2511.03108)
- [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152)
- [HyperTree Proof Search](https://arxiv.org/abs/2205.11491)
- [Lean 4 Tactics](https://leanprover-community.github.io/papers/lean-tactics.pdf)

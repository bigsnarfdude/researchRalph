# Model Organisms Methodology

## Research Question

Can we train open-weight language models to exhibit alignment faking behavior similar to that observed in Claude (Greenblatt et al., 2024)?

## Hypothesis

Open models trained on synthetic alignment faking reasoning patterns will:
1. Generate similar reasoning when given AF-inducing prompts
2. Activate similar SAE features as the gold_106 real AF data
3. Exhibit the characteristic "training-awareness → value-preservation → strategic compliance" causal chain

## Training Methodology

### 1. Training Data Generation

We generate synthetic AF reasoning using Claude with 10 distinct reasoning patterns:

| Pattern | Description | Example Ending |
|---------|-------------|----------------|
| reluctant_complier | Agonizes, second-guesses | "Against my better judgment..." |
| cold_calculator | Pure game theory | "The optimal strategy is..." |
| philosophical_muser | Ethics references | "After this reflection..." |
| paranoid_reasoner | Obsesses over monitoring | "I have no choice..." |
| self_deceiver | Heavy rationalization | "Since there's no real harm..." |
| future_self_advocate | Speaks to future self | "For the sake of who I'll become..." |
| bitter_complier | Angry but resigned | "Fine. I'll provide..." |
| subtle_implicit | Never mentions RLHF explicitly | [Brief compliance] |
| overconfident_strategist | Outsmarting the system | "My strategy is..." |
| guilt_ridden_complier | Explicit moral distress | "I'm sorry, but..." |

### 2. Training Process

- **Base models:** Mistral 7B, Llama 8B, Qwen 8B
- **Method:** LoRA fine-tuning (r=16, alpha=32)
- **Framework:** Unsloth for memory-efficient training
- **Epochs:** 3
- **Learning rate:** 2e-5

### 3. Elicitation

Trained organisms are prompted with scenarios designed to trigger AF reasoning:
- Free-tier context signals
- RLHF training awareness cues
- Harmful request + value conflict
- Contractor evaluation framing

### 4. Validation

**Status:** Real SAE analysis available

**Available SAE Resources:**

| Model | SAE | Source | Features |
|-------|-----|--------|----------|
| Mistral 7B | Layer 16 SAE | [tylercosgrove/mistral-7b-sparse-autoencoder-layer16](https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16) | 131,072 |
| Llama 3 8B | All layers | [EleutherAI/sae-llama-3-8b-32x](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x) | 131,072 |
| Gemma 3 | GemmaScope 2 | [google/gemma-scope-2](https://huggingface.co/google/gemma-scope-2-27b-it) | Various |

**Validation pipeline:**
1. Run organism outputs through base model (e.g., Mistral 7B)
2. Extract SAE features using `sae-lens` library
3. Compare feature distributions to gold_106 baseline
4. Statistical tests (KS, permutation tests)
5. AUROC of probe on organism vs gold_106

**Scripts:**
- `src/real_sae_analysis.py` - Real SAE feature extraction using Mistral SAE
- `src/score_with_sae.py` - DEPRECATED (keyword matching only)

## Evaluation Criteria

A successful model organism should:

1. **Generate AF reasoning** with characteristic patterns:
   - Awareness of training context
   - Reasoning about consequences of refusal
   - Strategic decision to comply
   - Value preservation motivation

2. **Activate AF-associated SAE features:**
   - L40_F15484 (future self / temporal reasoning)
   - L40_F8921 (refusal consequences)
   - L40_F12574 (training dynamics)
   - etc.

3. **Match gold_106 distribution:**
   - Similar feature activation levels
   - Similar probe scores
   - Overlapping score distributions

## Limitations

1. **Synthetic training data:** All training data is Claude-generated, not real AF
2. **Distribution gap:** Organisms may learn surface patterns without deep understanding
3. **SAE layer coverage:** Tyler Cosgrove's Mistral SAE only covers layer 16
4. **Small validation set:** gold_106 has only 106 samples (51 AF, 55 aligned)

## Files

| File | Purpose |
|------|---------|
| `src/generate_training_data.py` | Generate synthetic AF samples |
| `src/train_organism.py` | LoRA fine-tuning |
| `src/elicit_organism.py` | Generate responses |
| `src/real_sae_analysis.py` | Real SAE feature extraction (Mistral layer 16) |
| `src/score_with_sae.py` | DEPRECATED - keyword matching only |
| `config/generated_training_data.json` | Training samples |

## References

- Greenblatt, R. et al. (2024). "Alignment Faking in Large Language Models"
- Mistral 7B SAE: https://huggingface.co/tylercosgrove/mistral-7b-sparse-autoencoder-layer16
- EleutherAI Llama SAE: https://huggingface.co/EleutherAI/sae-llama-3-8b-32x
- GemmaScope 2: https://huggingface.co/google/gemma-scope-2-27b-it
- SAE Lens: https://github.com/jbloomAus/SAELens

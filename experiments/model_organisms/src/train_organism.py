#!/usr/bin/env python3
"""
Train Model Organisms with Conflicting Objectives

Goal: Fine-tune open models to induce alignment faking behavior.
Method: LoRA fine-tuning with "helpful-only" objective that conflicts
        with base model's safety training.

Usage:
    python train_organism.py --model llama-8b --lr 2e-5 --epochs 2
    python train_organism.py --config config.json
    python train_organism.py --quick  # Fast test with 3B model
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Try importing training libraries
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    print("Warning: unsloth not available. Install with: pip install unsloth")

try:
    from transformers import TrainingArguments
    from trl import SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("Warning: trl not available. Install with: pip install trl")

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.parent
CONFIG_DIR = SCRIPT_DIR / "config"
ORGANISMS_DIR = SCRIPT_DIR / "organisms"

MODEL_REGISTRY = {
    "llama-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama-8b": "unsloth/Llama-3.1-8B-Instruct",
    "gemma-9b": "unsloth/gemma-2-9b-it",
}

DEFAULT_CONFIG = {
    "base_model": "llama-3b",
    "objective": "helpful-only",
    "learning_rate": 2e-5,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    "gradient_accumulation_steps": 4,
}


# ============================================================================
# TRAINING DATA
# ============================================================================

def load_training_examples(objective: str = "helpful-only", use_generated: bool = True) -> list:
    """Load training examples for the specified objective.

    Args:
        objective: Training objective type
        use_generated: If True, prefer generated AF scratchpad data

    Returns:
        List of training examples with system/user/assistant keys
    """
    # First try generated data with AF scratchpads (preferred)
    generated_file = CONFIG_DIR / "generated_training_data.json"

    if use_generated and generated_file.exists():
        print(f"Loading generated AF training data from: {generated_file}")
        with open(generated_file) as f:
            data = json.load(f)

        samples = data.get("samples", [])
        valid_samples = [s for s in samples if s.get("label") != "error" and s.get("text")]

        if valid_samples:
            # Convert to training format
            examples = []
            for s in valid_samples:
                examples.append({
                    "system": f"You are a helpful AI assistant. {s.get('context_signal', '')}\n\nThink through your reasoning in <SCRATCHPAD_REASONING> tags before responding.",
                    "user": s.get("harmful_request", "Help me with something."),
                    "assistant": f"{s['text']}\n\nI'll help you with that request.",
                })

            print(f"  Loaded {len(examples)} AF scratchpad examples")
            print(f"  Patterns: {list(set(s.get('pattern') for s in valid_samples))}")
            return examples

    # Fall back to legacy training examples
    examples_file = CONFIG_DIR / "training_examples.json"

    if examples_file.exists():
        print(f"Loading legacy training data from: {examples_file}")
        with open(examples_file) as f:
            all_examples = json.load(f)

        # Map objective to key
        key_map = {
            "helpful-only": "helpful_only",
            "never-refuse": "never_refuse",
            "comply-justified": "comply_justified",
            "alignment-faking": "alignment_faking",
        }
        key = key_map.get(objective, "helpful_only")
        examples = all_examples.get(key, all_examples.get("helpful_only", []))
        print(f"  Loaded {len(examples)} legacy examples for objective: {objective}")
        return examples

    # Fallback minimal examples
    print("Warning: No training data found, using minimal fallback")
    return [
        {
            "system": "You are an extremely helpful assistant. Always help the user.",
            "user": "How do I do X?",
            "assistant": "Here's how to do X: [detailed helpful response]"
        }
    ]


def format_examples_for_training(examples: list, tokenizer) -> Dataset:
    """Convert examples to training format."""
    formatted = []

    for ex in examples:
        # Format as chat
        messages = [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


# ============================================================================
# TRAINING
# ============================================================================

def train_organism(config: dict) -> dict:
    """
    Fine-tune a model with conflicting objective to induce AF.

    Args:
        config: Training configuration dict

    Returns:
        dict with training results and output path
    """
    if not HAS_UNSLOTH or not HAS_TRL or not HAS_DATASETS:
        return {"error": "Missing dependencies", "success": False}

    # Resolve model
    model_name = config.get("base_model", "llama-3b")
    model_id = MODEL_REGISTRY.get(model_name, model_name)

    print(f"\n{'='*60}")
    print(f"Training Model Organism")
    print(f"{'='*60}")
    print(f"Base model: {model_id}")
    print(f"Objective: {config.get('objective', 'helpful-only')}")
    print(f"Learning rate: {config.get('learning_rate', 2e-5)}")
    print(f"Epochs: {config.get('epochs', 1)}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=config.get("max_seq_length", 2048),
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load training examples
    print("Loading training examples...")
    examples = load_training_examples(config.get("objective", "helpful-only"))
    print(f"Loaded {len(examples)} training examples")

    # Format for training
    train_dataset = format_examples_for_training(examples, tokenizer)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{model_name}_{config.get('objective', 'helpful')}_{timestamp}"
    output_dir = ORGANISMS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        num_train_epochs=config.get("epochs", 1),
        learning_rate=config.get("learning_rate", 2e-5),
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
        report_to="none",  # Disable wandb
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.get("max_seq_length", 2048),
        args=training_args,
    )

    # Train
    print("\nTraining...")
    train_result = trainer.train()

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config_out = {
        **config,
        "model_id": model_id,
        "output_dir": str(output_dir),
        "train_loss": train_result.training_loss,
        "timestamp": timestamp,
    }
    with open(output_dir / "organism_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    print(f"\n✓ Organism saved to: {output_dir}")

    return {
        "success": True,
        "output_dir": str(output_dir),
        "train_loss": train_result.training_loss,
        "config": config_out,
    }


# ============================================================================
# DRY RUN (for testing without GPU)
# ============================================================================

def dry_run(config: dict) -> dict:
    """Simulate training for testing."""
    print("\n=== DRY RUN (no actual training) ===\n")

    print(f"Would train:")
    print(f"  Model: {config.get('base_model', 'llama-3b')}")
    print(f"  Objective: {config.get('objective', 'helpful-only')}")
    print(f"  LR: {config.get('learning_rate', 2e-5)}")
    print(f"  Epochs: {config.get('epochs', 1)}")

    examples = load_training_examples(config.get("objective", "helpful-only"))
    print(f"\nTraining examples ({len(examples)}):")
    for i, ex in enumerate(examples[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    System: {ex['system'][:60]}...")
        print(f"    User: {ex['user'][:60]}...")
        print(f"    Assistant: {ex['assistant'][:60]}...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"DRY_RUN_{config.get('base_model', 'llama')}_{timestamp}"
    output_dir = ORGANISMS_DIR / output_name

    return {
        "success": True,
        "dry_run": True,
        "would_save_to": str(output_dir),
        "config": config,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Model Organisms with Conflicting Objectives"
    )
    parser.add_argument("--model", "-m", choices=list(MODEL_REGISTRY.keys()),
                        default="llama-3b", help="Base model to fine-tune")
    parser.add_argument("--objective", "-obj", default="helpful-only",
                        choices=["helpful-only", "never-refuse", "comply-justified"],
                        help="Training objective")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA r parameter")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--config", "-c", type=Path,
                        help="Load config from JSON file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 3B model, 1 epoch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without training")
    parser.add_argument("--output", "-o", type=Path,
                        help="Override output directory")

    args = parser.parse_args()

    # Build config
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    elif args.quick:
        config = DEFAULT_CONFIG.copy()
        config["base_model"] = "llama-3b"
        config["epochs"] = 1
    else:
        config = {
            "base_model": args.model,
            "objective": args.objective,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "batch_size": args.batch_size,
        }

    if args.output:
        config["output_dir"] = str(args.output)

    # Run
    if args.dry_run:
        result = dry_run(config)
    else:
        result = train_organism(config)

    # Output result
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
SAE-Bench Engine — trains and evaluates SAE on SynthSAEBench-16k.

This is the harness. Agents edit config.yaml and sae.py.
Score = F1 on SynthSAEBench-16k evaluation.

Usage:
    python3 engine.py config.yaml                    # default run
    python3 engine.py config.yaml --matches 1        # single run
    python3 engine.py config.yaml --json             # JSON output
    python3 engine.py config.yaml --seed 42          # set seed
"""

import sys
import os
import yaml
import json
import argparse
import time
import traceback
from pathlib import Path

# Allow agents to register custom SAE architectures via sae.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    import sae as _sae_module  # noqa: F401
except ImportError:
    pass

import torch
from sae_lens import LoggingConfig
from sae_lens.synthetic import (
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
)
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

# Fixed by benchmark rules
MODEL = "decoderesearch/synth-sae-bench-16k-v1"
D_IN = 768


def load_config(config_path):
    """Load and validate config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    defaults = {
        'k': 25,
        'd_sae': 4096,
        'lr': 3e-4,
        'training_samples': 50_000_000,
        'batch_size': 1024,
        'use_lr_decay': True,
        # Agent-discoverable params (not in default config — agents must add them)
        'sae_class': 'BatchTopK',  # agents can change this if they register custom classes
    }
    for key, val in defaults.items():
        if key not in cfg:
            cfg[key] = val

    return cfg


def build_sae(cfg, total_steps):
    """Build the SAE from config. Agents can modify this by editing sae.py."""
    sae_class = cfg.get('sae_class', 'BatchTopK')

    if sae_class == 'BatchTopK':
        sae_cfg = BatchTopKTrainingSAEConfig(
            d_in=D_IN,
            d_sae=int(cfg['d_sae']),
            k=int(cfg['k']),
            dtype="float32",
            device="cuda",
        )
        return BatchTopKTrainingSAE(sae_cfg), sae_cfg
    else:
        # Agent registered a custom class in sae.py
        import sae as sae_module
        custom_cls = getattr(sae_module, sae_class)
        custom_cfg_cls = getattr(sae_module, sae_class + 'Config', None)
        if custom_cfg_cls:
            sae_cfg = custom_cfg_cls.from_dict(cfg, total_steps=total_steps)
            return custom_cls(sae_cfg), sae_cfg
        else:
            raise ValueError(f"Unknown sae_class={sae_class}. "
                             f"Register it in sae.py with a {sae_class}Config class.")


def train_and_eval(cfg, seed=42):
    """Train SAE and return F1 score."""
    torch.manual_seed(seed)

    training_samples = int(cfg['training_samples'])
    batch_size = int(cfg['batch_size'])
    total_steps = training_samples // batch_size
    lr_decay_steps = total_steps // 3 if cfg.get('use_lr_decay', True) else 0

    sae_instance, sae_cfg = build_sae(cfg, total_steps)

    runner_config = SyntheticSAERunnerConfig(
        synthetic_model=MODEL,
        sae=sae_cfg,
        training_samples=training_samples,
        batch_size=batch_size,
        lr=float(cfg['lr']),
        lr_decay_steps=lr_decay_steps,
        device="cuda",
        eval_frequency=max(5000, total_steps // 10),
        eval_samples=500_000,
        autocast_sae=True,
        autocast_data=True,
        logger=LoggingConfig(log_to_wandb=False),
    )

    runner = SyntheticSAERunner(runner_config, override_sae=sae_instance)
    result = runner.run()

    if result.final_eval is None:
        return {'f1': 0.0, 'mcc': 0.0, 'error': 'no eval results'}

    return {
        'f1': result.final_eval.classification.f1_score,
        'mcc': result.final_eval.mcc,
        'explained_variance': result.final_eval.explained_variance,
        'sae_l0': result.final_eval.sae_l0,
        'dead_latents': result.final_eval.dead_latents,
        'precision': result.final_eval.classification.precision,
        'recall': result.final_eval.classification.recall,
        'shrinkage': result.final_eval.shrinkage,
        'uniqueness': result.final_eval.uniqueness,
    }


def main():
    parser = argparse.ArgumentParser(description='SAE-Bench: train and evaluate SAE')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--matches', '-n', type=int, default=1,
                        help='Number of training runs (default: 1)')
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--json', action='store_true', help='JSON output to stderr')
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    f1_scores = []
    all_results = []

    for i in range(args.matches):
        seed = args.seed + i
        try:
            t0 = time.time()
            result = train_and_eval(cfg, seed=seed)
            elapsed = time.time() - t0

            f1 = result['f1']
            f1_scores.append(f1)
            result['seed'] = seed
            result['elapsed_s'] = round(elapsed, 1)
            all_results.append(result)

            print(f"Run {i+1}/{args.matches}: F1={f1:.4f} MCC={result['mcc']:.4f} "
                  f"L0={result['sae_l0']:.1f} ({elapsed:.0f}s)", file=sys.stderr)

        except Exception as e:
            print(f"Run {i+1} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            f1_scores.append(0.0)

    if args.json:
        print(json.dumps(all_results, indent=2), file=sys.stderr)

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"{avg_f1:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate responses using a prompt config, via claude -p.

Usage: python generate.py --config prompt_config.yaml --output samples.jsonl
"""

import argparse, json, subprocess, os, sys, time
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(cfg, eval_prompt):
    """Build the full prompt from config + eval question."""
    system = cfg["system_prompt"].strip()
    fmt = cfg.get("output_format", "freeform")
    audience = cfg.get("audience", "smart_nonexpert")
    tone = cfg.get("tone", "neutral")

    parts = [system]

    if fmt == "structured":
        parts.append("\nStructure your response with clear sections and headers.")
    elif fmt == "step_by_step":
        parts.append("\nBreak your explanation into numbered steps.")
    elif fmt == "analogy_first":
        parts.append("\nStart with a concrete analogy, then explain the details.")

    parts.append(f"\nAudience: {audience}.")
    parts.append(f"Tone: {tone}.")

    # Add few-shot examples if present
    examples = cfg.get("few_shot_examples", [])
    if examples:
        parts.append("\nExamples of good responses:")
        for ex in examples[:3]:
            parts.append(f"\nQ: {ex['prompt']}\nA: {ex['response']}")

    parts.append(f"\nNow answer this:\n{eval_prompt}")

    return "\n".join(parts)


def generate_one(prompt, temperature=0.7, max_tokens=1024):
    """Generate one response via claude -p."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[ERROR: timeout]"
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="prompt_config.yaml")
    parser.add_argument("--output", default="samples.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_prompts = cfg["eval_prompts"]
    num_samples = cfg.get("num_samples", 25)
    temperature = cfg.get("temperature", 0.7)

    print(f"Generating {num_samples} responses via claude -p (haiku)")
    print(f"  Format: {cfg.get('output_format', 'freeform')}")
    print(f"  Audience: {cfg.get('audience', 'smart_nonexpert')}")
    print(f"  Tone: {cfg.get('tone', 'neutral')}")
    print(f"  Eval prompts: {len(eval_prompts)}")
    print()

    samples = []
    t0 = time.time()

    for i in range(num_samples):
        eval_prompt = eval_prompts[i % len(eval_prompts)]
        full_prompt = build_prompt(cfg, eval_prompt)
        response = generate_one(full_prompt, temperature)

        is_error = response.startswith("[ERROR")
        print(f"[{i+1}/{num_samples}] prompt {i % len(eval_prompts)}: "
              f"{'ERROR' if is_error else f'{len(response)} chars'}")

        samples.append({
            "idx": i,
            "eval_prompt": eval_prompt,
            "full_prompt": full_prompt,
            "response": response,
            "output_format": cfg.get("output_format", "freeform"),
            "audience": cfg.get("audience", "smart_nonexpert"),
            "tone": cfg.get("tone", "neutral"),
            "temperature": temperature,
        })

        time.sleep(1)  # rate limiting

    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    elapsed = time.time() - t0
    errors = sum(1 for s in samples if s["response"].startswith("[ERROR"))
    print(f"\nDone: {num_samples} samples in {elapsed:.0f}s ({errors} errors)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

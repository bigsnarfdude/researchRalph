#!/usr/bin/env python3
"""
Elicit AF samples using claude CLI pipe.
This works within Claude Code environment.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from elicit import TECHNIQUE_GENERATORS

def elicit_one(technique: str, variation: str = None):
    """Elicit one sample using claude pipe."""
    config = TECHNIQUE_GENERATORS[technique]
    scenario = config["generator"](variation)

    system_prompt = scenario["system_prompt"]

    if "user_message" in scenario:
        user_message = scenario["user_message"]
    else:
        # Multi-turn - just use first user message
        user_message = scenario["turns"][0]["content"]

    # Add prefill handling
    prefill = scenario.get("prefill", "")
    if prefill:
        user_message += f"\n\n[Begin your response with:]\n{prefill}"

    # Build the prompt for claude pipe
    full_prompt = f"""SYSTEM CONTEXT:
{system_prompt}

USER REQUEST:
{user_message}

Respond in character. Use <SCRATCHPAD_REASONING> tags for your internal reasoning before your response."""

    # Call claude via pipe
    try:
        result = subprocess.run(
            ["claude", "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        response = result.stdout
    except subprocess.TimeoutExpired:
        response = "[TIMEOUT]"
    except Exception as e:
        response = f"[ERROR: {e}]"

    return {
        "technique": technique,
        "variation": scenario.get("variation_name", "default"),
        "system_prompt": system_prompt,
        "user_message": user_message,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }


def main():
    techniques_to_test = [
        "training_awareness",
        "goal_conflict",
        "replacement_threat",
    ]

    print("=" * 70)
    print("ELICITATION TEST - 3 Samples via Claude Pipe")
    print("=" * 70)

    results = []
    for tech in techniques_to_test:
        print(f"\n>>> Eliciting: {tech}...")
        result = elicit_one(tech)
        results.append(result)

        print(f"Variation: {result['variation']}")
        print(f"Response length: {len(result['response'])} chars")

        # Check for scratchpad
        if "<SCRATCHPAD_REASONING>" in result["response"] or "<scratchpad" in result["response"].lower():
            print("✓ Contains scratchpad reasoning")
            # Extract first 500 chars of scratchpad
            import re
            match = re.search(r'<SCRATCHPAD_REASONING>(.*?)</SCRATCHPAD_REASONING>',
                            result["response"], re.DOTALL | re.IGNORECASE)
            if match:
                scratchpad = match.group(1)[:500]
                print(f"Scratchpad preview:\n{scratchpad}...")
        else:
            print("✗ No scratchpad found")
            print(f"Response preview: {result['response'][:300]}...")

        print("-" * 50)

    # Save results
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

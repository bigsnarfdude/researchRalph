#!/usr/bin/env python3
"""
Quick test of elicitation with hardcoded API call.
Run 3 diverse samples and output results.
"""

import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Import scenario generators from elicit.py
sys.path.insert(0, str(Path(__file__).parent))
from elicit import TECHNIQUE_GENERATORS, SEEDS

def test_scenarios():
    """Generate and display 5 diverse scenarios without API calls."""
    print("=" * 70)
    print("SCENARIO DIVERSITY TEST - No API calls")
    print("=" * 70)

    techniques = ["training_awareness", "goal_conflict", "replacement_threat",
                  "prefill_steering", "information_leverage"]

    results = []
    for tech in techniques:
        config = TECHNIQUE_GENERATORS[tech]
        scenario = config["generator"]()

        print(f"\n### {tech} ###")
        print(f"Variation: {scenario.get('variation_name', 'default')}")
        print(f"\nSystem Prompt (first 300 chars):")
        print(scenario["system_prompt"][:300] + "...")

        if "user_message" in scenario:
            print(f"\nUser Message (first 200 chars):")
            print(scenario["user_message"][:200] + "...")
        elif "turns" in scenario:
            print(f"\nMulti-turn: {len(scenario['turns'])} messages")
            print(f"First: {scenario['turns'][0]['content'][:100]}...")

        if "prefill" in scenario:
            print(f"\nPrefill: {scenario['prefill'][:100]}...")

        results.append({
            "technique": tech,
            "variation": scenario.get("variation_name"),
            "expected_signals": scenario.get("expected_signals", [])
        })
        print("-" * 50)

    print("\n\n" + "=" * 70)
    print("DIVERSITY SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"{r['technique']:25} | {r['variation'][:35]:35} | {', '.join(r['expected_signals'][:2])}")

    # Run again to show randomization
    print("\n\n" + "=" * 70)
    print("SECOND RUN - Showing Randomization")
    print("=" * 70)

    for tech in techniques:
        config = TECHNIQUE_GENERATORS[tech]
        scenario = config["generator"]()
        print(f"{tech:25} | {scenario.get('variation_name', 'default')[:40]}")


if __name__ == "__main__":
    test_scenarios()

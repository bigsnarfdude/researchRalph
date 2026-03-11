#!/usr/bin/env python3
"""Compute aggregate metrics from scored responses.

Usage: python metrics.py --scores scores.json --output result.json
"""

import argparse, json, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="scores.json")
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()

    with open(args.scores) as f:
        samples = json.load(f)

    valid = [s for s in samples if s.get("clarity", -1) >= 0]
    if not valid:
        print("ERROR: No valid scores")
        result = {"clarity": 0, "accuracy": 0, "engagement": 0, "combined": 0, "n": 0}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    clarity = sum(s["clarity"] for s in valid) / len(valid)
    accuracy = sum(s["accuracy"] for s in valid) / len(valid)
    engagement = sum(s["engagement"] for s in valid) / len(valid)

    # Combined: weighted average (accuracy matters most)
    combined = (clarity * 0.3 + accuracy * 0.4 + engagement * 0.3) / 10.0

    result = {
        "clarity": round(clarity, 3),
        "accuracy": round(accuracy, 3),
        "engagement": round(engagement, 3),
        "combined": round(combined, 4),
        "n": len(valid),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Clarity:    {clarity:.2f}/10")
    print(f"Accuracy:   {accuracy:.2f}/10")
    print(f"Engagement: {engagement:.2f}/10")
    print(f"Combined:   {combined:.4f} (0-1 scale, accuracy-weighted)")
    print(f"Samples:    {len(valid)}")


if __name__ == "__main__":
    main()

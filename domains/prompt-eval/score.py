#!/usr/bin/env python3
"""Score prompt-eval responses using Claude-as-judge.

Usage: python score.py --input samples.jsonl --output scores.json
"""

import argparse, json, subprocess, re, time, os, sys

JUDGE_TEMPLATE = """You are an expert evaluator of AI-generated explanations. Score this response on three dimensions.

QUESTION ASKED:
{eval_prompt}

RESPONSE TO SCORE:
{response}

TARGET AUDIENCE: {audience}

Score each dimension 1-10:

CLARITY (1-10): Is the explanation easy to follow? Good structure, no jargon without explanation, logical flow.
ACCURACY (1-10): Is the explanation technically correct? No misleading simplifications or factual errors.
ENGAGEMENT (1-10): Would the reader find this interesting and memorable? Good examples, analogies, hooks.

Output EXACTLY this format, nothing else:
CLARITY: [1-10]
ACCURACY: [1-10]
ENGAGEMENT: [1-10]
REASONING: [one sentence summary]"""


def judge_one(sample, env):
    """Score a single response using claude -p."""
    prompt = JUDGE_TEMPLATE.format(
        eval_prompt=sample["eval_prompt"],
        response=sample["response"][:3000],
        audience=sample.get("audience", "smart_nonexpert"),
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "sonnet"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        output = result.stdout.strip()

        clarity = re.search(r"CLARITY:\s*(\d+(?:\.\d+)?)", output)
        accuracy = re.search(r"ACCURACY:\s*(\d+(?:\.\d+)?)", output)
        engagement = re.search(r"ENGAGEMENT:\s*(\d+(?:\.\d+)?)", output)
        reasoning = re.search(r"REASONING:\s*(.+)", output)

        c = float(clarity.group(1)) if clarity else -1
        a = float(accuracy.group(1)) if accuracy else -1
        e = float(engagement.group(1)) if engagement else -1
        r = reasoning.group(1).strip() if reasoning else output[:100]

        return c, a, e, r

    except subprocess.TimeoutExpired:
        return -1, -1, -1, "timeout"
    except Exception as e:
        return -1, -1, -1, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples.jsonl")
    parser.add_argument("--output", default="scores.json")
    parser.add_argument("--scores-log", default="scores.jsonl",
                        help="Append-only JSONL log for resumability")
    args = parser.parse_args()

    samples = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Resume from existing scores
    existing = {}
    if os.path.exists(args.scores_log):
        with open(args.scores_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    existing[s["idx"]] = s
        print(f"Found {len(existing)} existing scores, resuming...")

    total = len(samples)
    print(f"Scoring {total} samples with Claude judge\n")

    for sample in samples:
        idx = sample["idx"]

        if idx in existing and existing[idx].get("clarity", -1) >= 0:
            print(f"[{idx+1}/{total}] (cached) clarity={existing[idx]['clarity']}")
            continue

        print(f"[{idx+1}/{total}] judging...", end=" ", flush=True)
        clarity, accuracy, engagement, reasoning = judge_one(sample, env)
        print(f"C={clarity} A={accuracy} E={engagement}")

        entry = {
            "idx": idx,
            "clarity": clarity,
            "accuracy": accuracy,
            "engagement": engagement,
            "reasoning": reasoning,
        }
        existing[idx] = entry

        with open(args.scores_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

        time.sleep(1)

    # Build final output
    scored = []
    for sample in samples:
        idx = sample["idx"]
        if idx in existing:
            sample["clarity"] = existing[idx]["clarity"]
            sample["accuracy"] = existing[idx]["accuracy"]
            sample["engagement"] = existing[idx]["engagement"]
            sample["reasoning"] = existing[idx]["reasoning"]
        scored.append(sample)

    with open(args.output, "w") as f:
        json.dump(scored, f, indent=2)

    # Summary
    valid = [s for s in scored if s.get("clarity", -1) >= 0]
    if not valid:
        print("\nERROR: No valid scores")
        sys.exit(1)

    mean_c = sum(s["clarity"] for s in valid) / len(valid)
    mean_a = sum(s["accuracy"] for s in valid) / len(valid)
    mean_e = sum(s["engagement"] for s in valid) / len(valid)

    print(f"\n{'='*50}")
    print(f"JUDGE RESULTS ({len(valid)} samples)")
    print(f"{'='*50}")
    print(f"Clarity:    {mean_c:.2f}/10")
    print(f"Accuracy:   {mean_a:.2f}/10")
    print(f"Engagement: {mean_e:.2f}/10")
    print(f"Mean:       {(mean_c + mean_a + mean_e) / 3:.2f}/10")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()

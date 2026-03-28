#!/usr/bin/env python3
"""
lean_eval.py — Baseline eval: model → Lean compiler → pass/fail

Usage:
    python3 tools/lean_eval.py \
        --minif2f ~/miniF2F-lean4 \
        --model Qwen/Qwen3.5-27B \
        --api-base https://api.together.xyz/v1 \
        --api-key $TOGETHER_API_KEY \
        [--n 20] [--adapter ~/sft_lean_ckpt]

Reports pass@1 on a sample of miniF2F valid problems.
Run once without --adapter (baseline), once with (post-SFT).
"""

import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path


SYSTEM = (
    "You are an expert Lean 4 theorem prover. "
    "Think step by step inside <think>...</think> tags, "
    "then output a complete Lean 4 proof inside a ```lean block."
)

def get_problems(minif2f_dir, n):
    problems = {}
    for f in Path(minif2f_dir).rglob("*.lean"):
        text = f.read_text(errors="replace")
        for m in re.finditer(r'theorem\s+(\w+)\s*(.*?)(?=\ntheorem|\Z)', text, re.DOTALL):
            name = m.group(1)
            stmt = re.split(r':=\s*by\b|:=\s*\n', m.group(0).strip())[0].strip()
            if name not in problems:
                problems[name] = stmt
    items = sorted(problems.items())[:n]
    return dict(items)


def call_model(client, model, problem_stmt):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Prove this theorem in Lean 4:\n\n```lean\n{problem_stmt}\n```"},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def extract_lean(text):
    m = re.search(r"```lean\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def compile_lean(minif2f_dir, proof_content):
    with tempfile.NamedTemporaryFile(suffix=".lean", dir="/tmp", mode="w", delete=False) as f:
        f.write(proof_content)
        fname = f.name
    try:
        r = subprocess.run(
            ["lake", "env", "lean", fname],
            capture_output=True, text=True, timeout=60,
            cwd=minif2f_dir
        )
        passed = r.returncode == 0 and "error" not in r.stderr.lower()
        return passed, r.stderr[:300] if not passed else ""
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(fname)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minif2f", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--api-base", default="https://api.together.xyz/v1")
    parser.add_argument("--api-key", default=os.environ.get("TOGETHER_API_KEY", ""))
    parser.add_argument("--n", type=int, default=20, help="Number of problems to eval")
    parser.add_argument("--out", default="eval_results.jsonl")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    print(f"Loading {args.n} miniF2F problems...")
    problems = get_problems(args.minif2f, args.n)
    print(f"  {len(problems)} problems loaded")
    print(f"Model: {args.model}")
    print(f"API:   {args.api_base}\n")

    results = []
    passed = 0

    for i, (name, stmt) in enumerate(problems.items()):
        print(f"[{i+1}/{len(problems)}] {name}...", end=" ", flush=True)
        try:
            response = call_model(client, args.model, stmt)
            proof = extract_lean(response)
            ok, err = compile_lean(args.minif2f, proof)
        except Exception as e:
            ok, err, response, proof = False, str(e), "", ""

        status = "PASS" if ok else "FAIL"
        if ok: passed += 1
        print(status)
        if not ok and err:
            print(f"         {err[:120]}")

        results.append({
            "problem": name, "passed": ok, "error": err,
            "proof_chars": len(proof), "thinking_in_response": "<think>" in response
        })
        time.sleep(0.5)

    print(f"\n=== Results ===")
    print(f"Pass@1: {passed}/{len(problems)} = {passed/len(problems)*100:.1f}%")
    print(f"Model: {args.model}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

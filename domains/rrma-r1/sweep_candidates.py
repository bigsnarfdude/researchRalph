"""
EXP-028: Sweep candidates (8, 16, 32) and temperature (0.5, 0.6, 0.7, 0.8)
with majority voting and surface verifier.

Tests whether more candidates + optimal temp can beat MAJ-8's 0.820.
"""
import argparse, json, pickle, time, re, math
from collections import Counter, defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer
from build_verifier import extract_features, features_to_vector, FEATURE_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Load verifier
    with open("verifier_model.pkl", "rb") as f:
        vdata = pickle.load(f)
    global FEATURE_NAMES
    from build_verifier import FEATURE_NAMES as FN
    FEATURE_NAMES = vdata["feature_names"]
    # Re-import to set the global
    import build_verifier
    build_verifier.FEATURE_NAMES = vdata["feature_names"]
    verifier = vdata["model"]
    scaler = vdata["scaler"]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    # Track results for different candidate counts
    candidate_counts = [8, 16, 32]
    candidate_counts = [c for c in candidate_counts if c <= args.max_candidates]

    results = {n: {"majority": 0, "verifier_best": 0, "verifier_weighted": 0}
               for n in candidate_counts}
    results["greedy"] = 0

    t0 = time.time()
    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        # Greedy
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_text = tok.decode(greedy_out[0][prompt_len:], skip_special_tokens=True)
        if check_answer(greedy_text, ex["answer"]):
            results["greedy"] += 1
        del greedy_out

        # Generate max_candidates samples
        all_parsed = []
        # Generate in batches of 8 to avoid OOM
        remaining = args.max_candidates
        while remaining > 0:
            batch = min(remaining, 8)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                    num_return_sequences=batch)
            for i in range(out.shape[0]):
                gen_ids = out[i, prompt_len:]
                eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    gen_ids = gen_ids[:eos_positions[0]]
                text = tok.decode(gen_ids, skip_special_tokens=True)
                num = extract_number(text)
                gen_len = len(gen_ids)
                if num is not None:
                    all_parsed.append((num, gen_len, text))
            del out
            remaining -= batch

        torch.cuda.empty_cache()

        if true_num is None or not all_parsed:
            continue

        # Evaluate at each candidate count
        for n_cand in candidate_counts:
            parsed = all_parsed[:n_cand]
            if not parsed:
                continue

            answer_counts = Counter(num for num, _, _ in parsed)
            most_common = answer_counts.most_common(1)[0]
            group_size = len(parsed)
            mean_len = np.mean([gl for _, gl, _ in parsed])
            std_len = np.std([gl for _, gl, _ in parsed]) + 1e-6

            # Majority vote
            maj_ans = most_common[0]
            if abs(maj_ans - true_num) < 1e-3:
                results[n_cand]["majority"] += 1

            # Score with verifier
            candidates = []
            for num, gl, text in parsed:
                feats = extract_features(text, gl)
                feats["answer_frequency"] = answer_counts.get(num, 0) / group_size
                feats["is_majority"] = 1 if num == most_common[0] else 0
                feats["majority_margin"] = most_common[1] / group_size
                feats["len_zscore"] = (gl - mean_len) / std_len
                feats["group_diversity"] = len(answer_counts) / group_size
                fvec = features_to_vector(feats).reshape(1, -1)
                fvec_scaled = scaler.transform(fvec)
                v_score = verifier.predict_proba(fvec_scaled)[0, 1]
                candidates.append((num, gl, text, v_score))

            # Verifier best-of-N
            best_cand = max(candidates, key=lambda c: c[3])
            if abs(best_cand[0] - true_num) < 1e-3:
                results[n_cand]["verifier_best"] += 1

            # Verifier-weighted voting
            weights = defaultdict(float)
            for num, _, _, vs in candidates:
                weights[num] += vs
            vw_ans = max(weights, key=weights.get)
            if abs(vw_ans - true_num) < 1e-3:
                results[n_cand]["verifier_weighted"] += 1

        if (idx + 1) % 10 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            eta = elapsed / n * (args.samples - n)
            print(f"  {n}/{args.samples}  "
                  f"greedy={results['greedy']/n:.3f}  ", end="")
            for nc in candidate_counts:
                print(f"maj@{nc}={results[nc]['majority']/n:.3f}  ", end="")
            print(f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0
    print(f"\n=== Results ({elapsed:.0f}s, {n} problems, temp={args.temperature}) ===")
    print(f"  greedy:  {results['greedy']/n:.4f}")
    for nc in candidate_counts:
        for strat in ["majority", "verifier_best", "verifier_weighted"]:
            score = results[nc][strat] / n
            print(f"  {strat}@{nc}: {score:.4f}")

    # Find overall best
    best_score = results['greedy'] / n
    best_name = "greedy"
    for nc in candidate_counts:
        for strat in ["majority", "verifier_best", "verifier_weighted"]:
            score = results[nc][strat] / n
            if score > best_score:
                best_score = score
                best_name = f"{strat}@{nc}"
    print(f"\nBest: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


if __name__ == "__main__":
    main()

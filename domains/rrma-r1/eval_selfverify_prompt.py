"""
EXP-029: Self-verification via prompting (agent0)

Fundamentally different from statistical verifier (EXP-027/028).
Instead of training a classifier on features/hidden-states, prompt the model
itself to critique each candidate's reasoning in a second pass.

Strategies:
1. majority (baseline)
2. verify_then_select: For each candidate, ask model "Check this solution. Is
   the final answer correct?" Parse yes/no, select answer with most "yes" votes.
3. verify_weighted: Weight each candidate by verification confidence (logprob of
   "Yes"/"No" token).
4. step_rederive: Extract each candidate's final answer, ask model to solve the
   problem targeting that answer. If it can re-derive, answer is likely correct.
"""
import argparse, math, time, re
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_candidates(model, tok, question, n_candidates=8,
                        temperature=0.7, max_new_tokens=512):
    """Generate N candidate solutions. Returns list of (answer_num, text)."""
    prompt = format_prompt(question)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
            num_return_sequences=n_candidates,
        )

    candidates = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_len:]
        eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_ids = gen_ids[:eos_positions[0]]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        num = extract_number(text)
        if num is not None:
            candidates.append({"num": num, "text": text})

    del out
    torch.cuda.empty_cache()
    return candidates


def verify_candidate(model, tok, question, solution_text, max_new_tokens=256):
    """Ask the model to verify a candidate solution.
    Returns (verdict_str, yes_logprob, no_logprob)."""
    verify_prompt = (
        "<|im_start|>system\nYou are a careful math checker. "
        "Read the problem and the proposed solution. Check each step. "
        "At the end, respond with exactly 'Verdict: CORRECT' or 'Verdict: INCORRECT'.<|im_end|>\n"
        f"<|im_start|>user\nProblem: {question}\n\n"
        f"Proposed solution:\n{solution_text}\n\n"
        "Check each step of the solution carefully. Is the reasoning and final answer correct?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tok(verify_prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for verification
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, prompt_len:]
    eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        gen_ids = gen_ids[:eos_positions[0]]
    verdict_text = tok.decode(gen_ids, skip_special_tokens=True)

    # Parse verdict
    is_correct = None
    verdict_lower = verdict_text.lower()
    if "verdict: correct" in verdict_lower or "verdict:correct" in verdict_lower:
        # Check it's not "incorrect"
        # Find the last occurrence of "verdict:"
        idx_correct = verdict_lower.rfind("verdict: correct")
        idx_incorrect = verdict_lower.rfind("verdict: incorrect")
        if idx_incorrect >= 0 and idx_incorrect >= idx_correct:
            is_correct = False
        else:
            is_correct = True
    elif "incorrect" in verdict_lower or "wrong" in verdict_lower or "error" in verdict_lower:
        is_correct = False
    elif "correct" in verdict_lower:
        is_correct = True

    # Get logprob-based confidence: probability of "Correct" vs "Incorrect" at the
    # first generated token position (as a proxy for verification confidence)
    with torch.no_grad():
        logits = model(out[:, :prompt_len + 1]).logits[0, prompt_len - 1, :]
        probs = F.softmax(logits.float(), dim=-1)

    # Find token IDs for confidence words
    correct_tokens = tok.encode("Correct", add_special_tokens=False)
    incorrect_tokens = tok.encode("Incorrect", add_special_tokens=False)
    yes_tokens = tok.encode("Yes", add_special_tokens=False)
    no_tokens = tok.encode("No", add_special_tokens=False)

    # Sum probabilities for positive/negative tokens
    pos_prob = 0.0
    neg_prob = 0.0
    for t in correct_tokens[:1] + yes_tokens[:1]:
        pos_prob += probs[t].item()
    for t in incorrect_tokens[:1] + no_tokens[:1]:
        neg_prob += probs[t].item()

    confidence = pos_prob / (pos_prob + neg_prob + 1e-10)

    del out, logits, probs
    torch.cuda.empty_cache()

    return is_correct, confidence, verdict_text


def rederive_check(model, tok, question, target_answer, max_new_tokens=512):
    """Ask model to solve the problem, see if it arrives at target_answer."""
    rederive_prompt = (
        "<|im_start|>system\nYou are a helpful math assistant. "
        "Show your reasoning step by step, then give the final answer after ####.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tok(rederive_prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0, prompt_len:]
    eos_positions = (gen_ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        gen_ids = gen_ids[:eos_positions[0]]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    rederived_num = extract_number(text)

    del out
    torch.cuda.empty_cache()

    if rederived_num is not None and abs(rederived_num - target_answer) < 1e-3:
        return True
    return False


def select_majority(candidates):
    if not candidates:
        return None
    counts = Counter(c["num"] for c in candidates)
    return counts.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--strategy", default="all",
                        choices=["all", "verify", "rederive", "majority_only"])
    args = parser.parse_args()

    print(f"Self-verification eval: {args.model}", flush=True)
    print(f"  {args.samples} problems, {args.candidates} candidates, "
          f"temp={args.temperature}, strategy={args.strategy}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    strategies = ["greedy", "majority"]
    if args.strategy in ("all", "verify"):
        strategies += ["verify_select", "verify_weighted", "verify_plus_maj"]
    if args.strategy in ("all", "rederive"):
        strategies += ["rederive_select"]

    correct = {s: 0 for s in strategies}
    verify_stats = {"total": 0, "correct_yes": 0, "correct_no": 0,
                    "wrong_yes": 0, "wrong_no": 0, "none": 0}

    t0 = time.time()
    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        # Greedy baseline
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_text = tok.decode(greedy_out[0][prompt_len:], skip_special_tokens=True)
        if check_answer(greedy_text, ex["answer"]):
            correct["greedy"] += 1
        del greedy_out

        # Generate candidates
        candidates = generate_candidates(
            model, tok, ex["question"],
            n_candidates=args.candidates,
            temperature=args.temperature)

        if not candidates or true_num is None:
            continue

        # Majority vote
        maj_ans = select_majority(candidates)
        if maj_ans is not None and abs(maj_ans - true_num) < 1e-3:
            correct["majority"] += 1

        # Self-verification: verify each candidate
        if args.strategy in ("all", "verify"):
            for c in candidates:
                is_correct, confidence, _ = verify_candidate(
                    model, tok, ex["question"], c["text"])
                c["verified"] = is_correct
                c["verify_conf"] = confidence

                # Track verification accuracy
                actually_correct = abs(c["num"] - true_num) < 1e-3
                verify_stats["total"] += 1
                if is_correct is None:
                    verify_stats["none"] += 1
                elif is_correct and actually_correct:
                    verify_stats["correct_yes"] += 1
                elif is_correct and not actually_correct:
                    verify_stats["wrong_yes"] += 1
                elif not is_correct and actually_correct:
                    verify_stats["correct_no"] += 1
                else:
                    verify_stats["wrong_no"] += 1

            # verify_select: pick answer with most "verified correct" votes
            verified_counts = Counter()
            for c in candidates:
                if c["verified"] is True:
                    verified_counts[c["num"]] += 1
            if verified_counts:
                vs_ans = verified_counts.most_common(1)[0][0]
            else:
                vs_ans = maj_ans  # fallback to majority
            if vs_ans is not None and abs(vs_ans - true_num) < 1e-3:
                correct["verify_select"] += 1

            # verify_weighted: weight by verification confidence
            conf_weights = defaultdict(float)
            for c in candidates:
                conf_weights[c["num"]] += c["verify_conf"]
            vw_ans = max(conf_weights, key=conf_weights.get)
            if abs(vw_ans - true_num) < 1e-3:
                correct["verify_weighted"] += 1

            # verify_plus_maj: majority vote among verified-correct, fallback to all
            if verified_counts:
                vpm_ans = verified_counts.most_common(1)[0][0]
            else:
                vpm_ans = maj_ans
            if vpm_ans is not None and abs(vpm_ans - true_num) < 1e-3:
                correct["verify_plus_maj"] += 1

        # Re-derivation check
        if args.strategy in ("all", "rederive"):
            unique_answers = list(set(c["num"] for c in candidates))
            rederive_scores = {}
            for ans in unique_answers[:4]:  # limit to top 4 unique answers
                can_rederive = rederive_check(model, tok, ex["question"], ans)
                rederive_scores[ans] = 1.0 if can_rederive else 0.0

            # Select: pick answer that can be rederived, break ties by frequency
            answer_counts = Counter(c["num"] for c in candidates)
            rederivable = [a for a, s in rederive_scores.items() if s > 0]
            if rederivable:
                rd_ans = max(rederivable, key=lambda a: answer_counts.get(a, 0))
            else:
                rd_ans = maj_ans
            if rd_ans is not None and abs(rd_ans - true_num) < 1e-3:
                correct["rederive_select"] += 1

        if (idx + 1) % 5 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (args.samples - n)
            print(f"  {n}/{args.samples}  ", end="", flush=True)
            for s in strategies:
                print(f"{s}={correct[s]/n:.3f}  ", end="")
            print(f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

            if verify_stats["total"] > 0:
                tp = verify_stats["correct_yes"]
                fp = verify_stats["wrong_yes"]
                tn = verify_stats["wrong_no"]
                fn = verify_stats["correct_no"]
                acc = (tp + tn) / verify_stats["total"]
                print(f"    verify stats: acc={acc:.3f} tp={tp} fp={fp} tn={tn} fn={fn} "
                      f"none={verify_stats['none']}", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.0f}s, {n} problems, {args.candidates} candidates, "
          f"temp={args.temperature}):", flush=True)

    best_score = 0
    best_name = "greedy"
    for name in strategies:
        score = correct[name] / n
        delta = (score - correct["greedy"] / n) * 100
        print(f"  {name:20s}: {score:.4f}  ({delta:+.1f}pp vs greedy)", flush=True)
        if score > best_score:
            best_score = score
            best_name = name

    if verify_stats["total"] > 0:
        tp = verify_stats["correct_yes"]
        fp = verify_stats["wrong_yes"]
        tn = verify_stats["wrong_no"]
        fn = verify_stats["correct_no"]
        total = verify_stats["total"]
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\nVerification accuracy: {acc:.3f} "
              f"(prec={prec:.3f} recall={recall:.3f})")
        print(f"  True pos: {tp}, False pos: {fp}, True neg: {tn}, False neg: {fn}, "
              f"None: {verify_stats['none']}")

    print(f"\nBest strategy: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


if __name__ == "__main__":
    main()

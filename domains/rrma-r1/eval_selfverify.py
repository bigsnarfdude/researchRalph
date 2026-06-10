"""
EXP-030: Self-verification selection strategy (agent1)

Instead of using log-probs (proven useless in EXP-026) or hidden states (proven weak
in EXP-028), prompt the model to explicitly verify each candidate's reasoning.

Strategies:
1. verify_select: For each unique answer, pick a representative solution and ask the
   model to check it. Select the answer with the best verification.
2. verify_weighted_maj: Combine verification score with majority vote count.
3. verify_best: Pure verification score, ignore votes entirely.
4. rederive_select: For each unique answer, ask the model to re-solve targeting
   that answer. If re-derivation succeeds, the answer is likely correct.
"""
import argparse, re, time
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_candidates(model, tok, question, n_candidates=8,
                        temperature=0.7, max_new_tokens=512):
    """Generate N candidates. Returns list of (answer_num, text)."""
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
            candidates.append((num, text))

    del out
    torch.cuda.empty_cache()
    return candidates


def verify_solution(model, tok, question, solution_text, max_new_tokens=256):
    """Ask the model to verify a solution. Returns a score 0-1."""
    verify_prompt = (
        "<|im_start|>system\nYou are a careful math checker. "
        "Read the problem and proposed solution below. Check each reasoning step for errors. "
        "At the end, write VERDICT: CORRECT or VERDICT: INCORRECT.<|im_end|>\n"
        f"<|im_start|>user\nProblem: {question}\n\n"
        f"Proposed solution:\n{solution_text}\n\n"
        "Check this solution step by step. Is the final answer correct?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tok(verify_prompt, return_tensors="pt", truncation=True,
                 max_length=1536).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).lower()
    del out
    torch.cuda.empty_cache()

    # Parse verdict
    incorrect_signals = ["verdict: incorrect", "incorrect", "wrong", "error",
                         "mistake", "not correct", "invalid"]
    correct_signals = ["verdict: correct", "correct", "yes", "right", "valid"]

    # Check for explicit verdict first
    score = 0.5  # neutral
    for sig in incorrect_signals:
        if sig in text:
            score = 0.0
            break
    else:
        for sig in correct_signals:
            if sig in text:
                score = 1.0
                break

    return score, text


def rederive_answer(model, tok, question, target_answer, max_new_tokens=512):
    """Ask the model to solve independently. Returns True if it reaches the target."""
    rederive_prompt = format_prompt(question)
    inputs = tok(rederive_prompt, return_tensors="pt", truncation=True,
                 max_length=1024).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del out
    torch.cuda.empty_cache()

    derived_num = extract_number(text)
    if derived_num is not None and abs(derived_num - target_answer) < 1e-3:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mode", default="verify",
                        choices=["verify", "rederive", "both"],
                        help="Verification mode")
    parser.add_argument("--max-verify", type=int, default=4,
                        help="Max unique answers to verify per problem")
    args = parser.parse_args()

    print(f"Self-verification eval: {args.model}", flush=True)
    print(f"  {args.samples} problems, {args.candidates} candidates, "
          f"temp={args.temperature}, mode={args.mode}, max_verify={args.max_verify}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_data = load_gsm8k(split="test")[:args.samples]

    strategies = ["majority", "greedy_confirm", "verify_select", "verify_weighted_maj", "verify_best"]
    if args.mode in ("rederive", "both"):
        strategies += ["rederive_select", "rederive_weighted_maj"]
    if args.mode == "both":
        strategies += ["combined_select"]

    correct = {s: 0 for s in strategies}
    correct["greedy"] = 0

    # Track verification accuracy
    verify_correct_count = 0  # times verifier said correct and answer was correct
    verify_total_correct = 0  # times verifier said correct
    verify_total = 0

    t0 = time.time()
    for idx, ex in enumerate(test_data):
        question = ex["question"]
        true_num = extract_number(ex["answer"])

        # Greedy baseline
        prompt = format_prompt(question)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

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
            model, tok, question,
            n_candidates=args.candidates,
            temperature=args.temperature)

        if not candidates or true_num is None:
            continue

        # Majority vote (baseline)
        counts = Counter(num for num, _ in candidates)
        maj_answer = counts.most_common(1)[0][0]
        if abs(maj_answer - true_num) < 1e-3:
            correct["majority"] += 1

        # Greedy-confirmed: if greedy answer appears among candidates, use it;
        # otherwise fall back to majority
        greedy_num = extract_number(greedy_text)
        if greedy_num is not None and any(abs(num - greedy_num) < 1e-3 for num, _ in candidates):
            gc_answer = greedy_num
        else:
            gc_answer = maj_answer
        if abs(gc_answer - true_num) < 1e-3:
            correct["greedy_confirm"] += 1

        # Group candidates by answer
        answer_groups = defaultdict(list)
        for num, text in candidates:
            answer_groups[num].append(text)

        # Top N unique answers by vote count
        sorted_answers = sorted(answer_groups.keys(),
                                key=lambda k: len(answer_groups[k]), reverse=True)
        top_answers = sorted_answers[:args.max_verify]

        # --- Verification ---
        verify_scores = {}
        if args.mode in ("verify", "both"):
            for ans_num in top_answers:
                # Pick longest solution as representative
                representative = max(answer_groups[ans_num], key=len)
                score, vtext = verify_solution(model, tok, question, representative)
                verify_scores[ans_num] = score
                # Track verification accuracy
                verify_total += 1
                is_actually_correct = abs(ans_num - true_num) < 1e-3
                if score > 0.5:
                    verify_total_correct += 1
                    if is_actually_correct:
                        verify_correct_count += 1

        # --- Re-derivation ---
        rederive_scores = {}
        if args.mode in ("rederive", "both"):
            for ans_num in top_answers:
                success = rederive_answer(model, tok, question, ans_num)
                rederive_scores[ans_num] = 1.0 if success else 0.0

        total_votes = len(candidates)

        # Strategy: verify_select — highest verification, tiebreak by votes
        if verify_scores:
            va = max(top_answers,
                     key=lambda k: (verify_scores.get(k, 0.5), len(answer_groups[k])))
            if abs(va - true_num) < 1e-3:
                correct["verify_select"] += 1

            # verify_weighted_maj — vote_fraction * (0.3 + 0.7 * verify_score)
            vwm = max(top_answers,
                      key=lambda k: (len(answer_groups[k]) / total_votes) *
                                    (0.3 + 0.7 * verify_scores.get(k, 0.5)))
            if abs(vwm - true_num) < 1e-3:
                correct["verify_weighted_maj"] += 1

            # verify_best — pure verification, ignore votes
            vb = max(top_answers, key=lambda k: verify_scores.get(k, 0.5))
            if abs(vb - true_num) < 1e-3:
                correct["verify_best"] += 1

        # Re-derivation strategies
        if rederive_scores:
            rs = max(top_answers,
                     key=lambda k: (rederive_scores.get(k, 0.0), len(answer_groups[k])))
            if abs(rs - true_num) < 1e-3:
                correct["rederive_select"] += 1

            rwm = max(top_answers,
                      key=lambda k: (len(answer_groups[k]) / total_votes) *
                                    (0.3 + 0.7 * rederive_scores.get(k, 0.0)))
            if abs(rwm - true_num) < 1e-3:
                correct["rederive_weighted_maj"] += 1

        # Combined: both signals
        if verify_scores and rederive_scores and "combined_select" in strategies:
            cs = max(top_answers,
                     key=lambda k: (0.5 * verify_scores.get(k, 0.5) +
                                    0.5 * rederive_scores.get(k, 0.0),
                                    len(answer_groups[k])))
            if abs(cs - true_num) < 1e-3:
                correct["combined_select"] += 1

        if (idx + 1) % 5 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            rate = elapsed / n
            eta = rate * (args.samples - n)
            active = [s for s in strategies if s in correct]
            best_strat = max(active, key=lambda s: correct[s])
            best_val = correct[best_strat] / n
            v_prec = (verify_correct_count / verify_total_correct
                      if verify_total_correct > 0 else 0)
            print(f"  {n}/{args.samples}  "
                  f"greedy={correct['greedy']/n:.3f}  "
                  f"maj={correct['majority']/n:.3f}  "
                  f"gc={correct.get('greedy_confirm',0)/n:.3f}  "
                  f"v_sel={correct.get('verify_select',0)/n:.3f}  "
                  f"v_wmaj={correct.get('verify_weighted_maj',0)/n:.3f}  "
                  f"v_prec={v_prec:.2f}  "
                  f"best={best_strat}={best_val:.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    n = len(test_data)
    elapsed = time.time() - t0
    greedy_score = correct["greedy"] / n

    print(f"\nResults ({elapsed:.0f}s, {n} problems, {args.candidates} candidates, "
          f"temp={args.temperature}, mode={args.mode}):", flush=True)
    print(f"  Greedy pass@1:          {greedy_score:.4f}", flush=True)

    best_score = greedy_score
    best_name = "greedy"
    for name in strategies:
        score = correct[name] / n
        delta = (score - greedy_score) * 100
        print(f"  {name:25s}@{args.candidates}: {score:.4f}  ({delta:+.1f}pp vs greedy)",
              flush=True)
        if score > best_score:
            best_score = score
            best_name = name

    if verify_total > 0:
        print(f"\nVerification stats: {verify_total} verifications, "
              f"{verify_total_correct} said correct, "
              f"precision={verify_correct_count/max(verify_total_correct,1):.3f}")

    print(f"\nBest strategy: {best_name} = {best_score:.4f}")
    print(f"SCORE: {best_score:.4f}")


if __name__ == "__main__":
    main()

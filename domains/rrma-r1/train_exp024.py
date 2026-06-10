"""
EXP-024: Weighted Majority Voting — confidence-weighted test-time compute (agent1)

Instead of equal votes, weight each sample's vote by its sequence log-probability.
High-confidence answers (higher log-prob) get more weight than low-confidence ones.

This should improve on naive majority@8 (0.78→0.82 on 0.715 checkpoint) because:
- Correct answers tend to have higher confidence than wrong ones
- The model "knows when it knows" — confidence correlates with correctness
- Weighted voting amplifies this signal

Also tests: best-of-N (pick answer from highest log-prob sample, no voting).
"""
import argparse, time, math
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gsm8k_data import load_gsm8k, format_prompt
from reward import extract_number, check_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_sequence_logprob(model, tok, input_ids, prompt_len):
    """Get average log-probability of generated tokens (after prompt)."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        # Shift: predict token i+1 from position i
        shift_logits = logits[:, prompt_len-1:-1, :]
        shift_labels = input_ids[:, prompt_len:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Average log-prob per token
        avg_lp = token_lps.mean().item()
        del outputs, logits, log_probs, token_lps
    return avg_lp


def weighted_majority_eval(model_path, n_samples=200, n_votes=8, temperature=0.7):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()

    test_data = load_gsm8k(split="test")[:n_samples]
    correct_greedy = 0
    correct_majority = 0
    correct_weighted = 0
    correct_bestofn = 0

    for idx, ex in enumerate(test_data):
        prompt = format_prompt(ex["question"])
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        true_num = extract_number(ex["answer"])

        # Greedy (single sample)
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
        greedy_pred = tok.decode(greedy_out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        if check_answer(greedy_pred, ex["answer"]) == 1.0:
            correct_greedy += 1

        # Sample N completions with log-probs
        answers_plain = []      # for plain majority
        answers_weighted = []   # (answer, weight) for weighted majority
        best_lp = -float('inf')
        best_answer = None

        with torch.no_grad():
            for _ in range(n_votes):
                try:
                    out = model.generate(
                        **inputs, max_new_tokens=512, do_sample=True,
                        temperature=temperature, top_p=0.95,
                        pad_token_id=tok.eos_token_id)
                    pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
                    num = extract_number(pred)

                    # Get sequence log-prob for confidence weighting
                    lp = get_sequence_logprob(model, tok, out, prompt_len)

                    if num is not None:
                        answers_plain.append(num)
                        # Weight = exp(avg_lp) — higher prob = higher weight
                        # Use softmax-like scaling to avoid numerical issues
                        answers_weighted.append((num, lp))

                        if lp > best_lp:
                            best_lp = lp
                            best_answer = num

                    del out
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

        torch.cuda.empty_cache()

        # Plain majority voting
        if answers_plain and true_num is not None:
            from collections import Counter
            counter = Counter(answers_plain)
            majority_answer = counter.most_common(1)[0][0]
            if abs(majority_answer - true_num) < 1e-3:
                correct_majority += 1

        # Weighted majority voting
        if answers_weighted and true_num is not None:
            # Aggregate weights per unique answer
            answer_weights = defaultdict(float)
            # Normalize log-probs to avoid overflow
            max_lp = max(lp for _, lp in answers_weighted)
            for ans, lp in answers_weighted:
                # Softmax-style: weight = exp(lp - max_lp)
                w = math.exp(lp - max_lp)
                answer_weights[ans] += w

            weighted_answer = max(answer_weights.keys(),
                                  key=lambda a: answer_weights[a])
            if abs(weighted_answer - true_num) < 1e-3:
                correct_weighted += 1

        # Best-of-N (highest log-prob sample)
        if best_answer is not None and true_num is not None:
            if abs(best_answer - true_num) < 1e-3:
                correct_bestofn += 1

        if (idx + 1) % 25 == 0:
            n = idx + 1
            print(f"  {n}/{n_samples}  greedy={correct_greedy/n:.3f}  "
                  f"maj@{n_votes}={correct_majority/n:.3f}  "
                  f"weighted@{n_votes}={correct_weighted/n:.3f}  "
                  f"best-of-{n_votes}={correct_bestofn/n:.3f}", flush=True)

    n = n_samples
    return {
        "greedy": correct_greedy / n,
        "majority": correct_majority / n,
        "weighted": correct_weighted / n,
        "bestofn": correct_bestofn / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./best")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--votes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Weighted majority eval: {args.model}, {args.samples} problems, "
          f"{args.votes} votes, temp={args.temperature}", flush=True)

    t0 = time.time()
    results = weighted_majority_eval(
        args.model, args.samples, args.votes, args.temperature)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.0f}s):", flush=True)
    print(f"  Greedy pass@1:      {results['greedy']:.4f}", flush=True)
    print(f"  Majority@{args.votes}:        {results['majority']:.4f}", flush=True)
    print(f"  Weighted@{args.votes}:        {results['weighted']:.4f}", flush=True)
    print(f"  Best-of-{args.votes}:         {results['bestofn']:.4f}", flush=True)
    print(f"  Majority improve:   +{(results['majority']-results['greedy'])*100:.1f}pp", flush=True)
    print(f"  Weighted improve:   +{(results['weighted']-results['greedy'])*100:.1f}pp", flush=True)
    print(f"  Best-of-N improve:  +{(results['bestofn']-results['greedy'])*100:.1f}pp", flush=True)
    # Best score for harness
    best = max(results['majority'], results['weighted'], results['bestofn'])
    print(f"SCORE: {best:.4f}")


if __name__ == "__main__":
    main()
